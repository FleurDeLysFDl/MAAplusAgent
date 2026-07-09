"""HelloAgents装配代码：把MAA-MCP-Server的能力接到ReActAgent里做自由探索。

hello_agents（pip安装的 hello-agents 包）本身没有现成的MCPTool，这里用mcp SDK的
stdio客户端自己桥接：启动maa_mcp_server.py子进程，握手拿到工具列表，把每个远程工具
包装成一个hello_agents.tools.base.Tool注册进ToolRegistry。
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from mcp.types import Tool as McpToolSpec

from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.circuit_breaker import CircuitBreaker
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_broadcaster import LogBroadcaster, attach_broadcaster  # noqa: E402
from react_agent_vision import install_vision_support  # noqa: E402

install_vision_support()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "maa_mcp_server.py"

DEFAULT_MAX_STEPS = 30
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8765

EXPLORATION_PROMPT_TEMPLATE = """
你正在自由探索手游【{display_name}】。
游戏特点：{exploration_goal_hint}

你的目标：
1. 尽量发现你还没见过的界面和功能
2. 遇到已知任务时优先调用 list_known_tasks / run_known_task，不要重复摸索
3. 点击(click)/滑动(swipe)前必须先调用 screenshot()，坐标必须来自screenshot()返回的OCR候选框，不要自己估算坐标
4. 需要关闭弹窗/浮层、返回上一级界面时，优先调用press_back（安卓系统返回键），不需要猜坐标；
   press_back无效时，或者要点没有文字、OCR识别不出来的图标类按钮（比如关闭×、返回箭头）时，
   调用get_raw_image看原图，确认目标实际像素坐标后用click_on_image点击，不要凭空估算坐标
5. 如果screenshot()返回结果里出现sensitive_warning，立即停止操作，调用Finish工具汇报情况
6. 不要进入战斗/关卡/副本类功能：看到"进入战斗"、"作战开始"等战斗相关按钮不要点击，
   识别到这是战斗/编队/关卡界面后直接返回或换个方向探索，不要摸索战斗内部的具体玩法

本次探索最多执行 {max_steps} 步。
"""


class MCPServerBridge:
    """在后台线程里维护一条到MAA-MCP-Server子进程的长连接

    ReActAgent/ToolRegistry都是同步调用，而mcp SDK的ClientSession是异步的，
    这里用一个专属的asyncio事件循环+后台线程做桥接，保证子进程只启动一次，
    safety_guard的坐标白名单、exploration memory等服务端状态在整个探索会话内保持连续。
    """

    def __init__(self, command: str, args: list[str], cwd: str | None = None):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.tools: list[McpToolSpec] = []

        future = asyncio.run_coroutine_threadsafe(self._connect(command, args, cwd), self._loop)
        future.result(timeout=60)

    async def _connect(self, command: str, args: list[str], cwd: str | None) -> None:
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(command=command, args=args, cwd=cwd)
        read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        listing = await self._session.list_tools()
        self.tools = listing.tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self._session is not None
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        result = future.result(timeout=120)

        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        if result.isError:
            raise RuntimeError("; ".join(texts) or f"MCP工具'{name}'调用失败")
        if not texts:
            return {}
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return {"text": texts[0]}

    def close(self) -> None:
        if self._exit_stack is not None:
            future = asyncio.run_coroutine_threadsafe(self._exit_stack.aclose(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _schema_to_parameters(schema: dict[str, Any] | None) -> list[ToolParameter]:
    schema = schema or {}
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    return [
        ToolParameter(
            name=name,
            type=prop.get("type", "string"),
            description=prop.get("description", name),
            required=name in required,
            default=prop.get("default"),
        )
        for name, prop in properties.items()
    ]


class MCPTool(Tool):
    """把MAA-MCP-Server的一个远程工具包装成hello_agents的Tool"""

    def __init__(self, bridge: MCPServerBridge, spec: McpToolSpec):
        super().__init__(name=spec.name, description=spec.description or spec.name)
        self._bridge = bridge
        self._parameters = _schema_to_parameters(spec.inputSchema)

    def get_parameters(self) -> list[ToolParameter]:
        return self._parameters

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        try:
            data = self._bridge.call_tool(self.name, parameters)
        except Exception as e:
            return ToolResponse.error(code=ToolErrorCode.EXECUTION_ERROR, message=str(e))

        # get_raw_image返回的是原始base64字符串（走call_tool的非JSON兜底分支包成
        # {"text": base64str}），这里直接原样透传，不要JSON包一层：
        # react_agent_vision.py的补丁靠result==纯base64字符串来识别"这是张图"，
        # 拿去拼image_url内容块给多模态LLM看，包一层JSON会让这个识别失效
        if self.name == "get_raw_image" and isinstance(data, dict) and "text" in data:
            return ToolResponse.success(text=data["text"], data=data)

        text = data.get("sensitive_warning") if isinstance(data, dict) else None
        if not text:
            text = json.dumps(data, ensure_ascii=False)
        return ToolResponse.success(text=text, data=data if isinstance(data, dict) else {"result": data})


def load_profile(profile_path: str | Path) -> dict[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_exploration_prompt(profile: dict[str, Any]) -> str:
    return EXPLORATION_PROMPT_TEMPLATE.format(
        display_name=profile["display_name"],
        exploration_goal_hint=profile["exploration_goal_hint"].strip(),
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )


def build_agent(profile_path: str | Path) -> tuple[ReActAgent, MCPServerBridge, LogBroadcaster]:
    profile = load_profile(profile_path)

    bridge = MCPServerBridge(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT), str(profile_path)],
        cwd=str(PROJECT_ROOT),
    )

    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    registry = ToolRegistry(circuit_breaker=circuit_breaker)
    for spec in bridge.tools:
        registry.register_tool(MCPTool(bridge, spec))

    llm = HelloAgentsLLM()
    agent = ReActAgent(
        name=f"{profile['game_id']}-explorer",
        llm=llm,
        tool_registry=registry,
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )

    dashboard_cfg = profile.get("dashboard", {}) or {}
    broadcaster = LogBroadcaster(
        host=dashboard_cfg.get("host", DEFAULT_DASHBOARD_HOST),
        port=dashboard_cfg.get("port", DEFAULT_DASHBOARD_PORT),
    )
    broadcaster.start()
    if agent.trace_logger is not None:
        attach_broadcaster(agent.trace_logger, broadcaster)

    return agent, bridge, broadcaster


def main() -> None:
    # Windows控制台默认GBK编码，hello_agents等依赖会print emoji/特殊字符，
    # 不reconfigure的话会在打印时直接UnicodeEncodeError崩掉整个进程。
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    if len(sys.argv) < 2:
        print("用法: python agent_runner.py <profile.yaml路径>", file=sys.stderr)
        sys.exit(1)

    profile_path = sys.argv[1]
    profile = load_profile(profile_path)
    agent, bridge, broadcaster = build_agent(profile_path)
    print(f"📡 实时日志面板: {broadcaster.url}")
    try:
        agent.run(build_exploration_prompt(profile))
    finally:
        bridge.close()
        broadcaster.stop()


if __name__ == "__main__":
    main()
