"""HelloAgents装配代码：把MAA-MCP-Server的能力接到ReActAgent里做自由探索。

hello_agents（pip安装的 hello-agents 包）本身没有现成的MCPTool，这里用mcp SDK的
stdio客户端自己桥接：启动maa_mcp_server.py子进程，握手拿到工具列表，把每个远程工具
包装成一个hello_agents.tools.base.Tool注册进ToolRegistry。
"""
from __future__ import annotations

import asyncio
import json
import os
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
from emulator import ensure_emulator_ready  # noqa: E402
from log_broadcaster import LogBroadcaster, attach_broadcaster  # noqa: E402
from react_agent_vision import install_vision_support  # noqa: E402

install_vision_support()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "maa_mcp_server.py"

DEFAULT_MAX_STEPS = 30
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8765

# 感知/操作层的规则，跟"为什么要做"（自由探索 or 用户指令）无关，两种prompt都复用同一份，
# 以后新增"接受用户指令"之类的模式时也是拼这份规则，不用重写一遍工具用法
AGENT_OPERATING_RULES = """
1. 点击(click)/滑动(swipe)前必须先调用 screenshot()，坐标必须来自screenshot()返回的OCR候选框，不要自己估算坐标
2. 需要关闭弹窗/浮层、返回上一级界面时，优先调用press_back（安卓系统返回键），不需要猜坐标；
   press_back无效时，或者要点没有文字、OCR识别不出来的图标类按钮（比如关闭×、返回箭头）时，
   调用get_raw_image看原图，确认目标实际像素坐标后用click_on_image点击，不要凭空估算坐标
3. 如果screenshot()返回结果里出现sensitive_warning，立即停止操作，调用Finish工具汇报情况
4. 不要进入战斗/关卡/副本类功能：看到"进入战斗"、"作战开始"等战斗相关按钮不要点击，
   识别到这是战斗/编队/关卡界面后直接返回或换个方向探索，不要摸索战斗内部的具体玩法
5. 遇到已知任务时优先调用 list_known_tasks / run_known_task，不要重复摸索
6. 如果screenshot()返回结果里known_ineffective不为空，说明这些操作之前在这个界面上
   试过没有效果（点了/按了界面没变），不要重复尝试，直接换别的方式
""".strip()

# known_actions/replay_action这条捷径只在"重复已知任务"场景（比如按用户指令执行）才给，
# 探索阶段不注册replay_action工具、也不提这条规则——探索的目的就是发现新东西，如果碰到
# 已知节点就走捷径直接重放，很容易在几个互通的已知节点之间来回打转，反而探索不出新内容
# （实测出现过"预设列表↔创建界面"来回重放两个已知动作、不再往前推进的情况）
KNOWN_ACTIONS_RULE = """
7. 如果screenshot()返回结果里known_actions不为空，说明这个界面之前来过、已经有验证过
   坐标的操作，优先从known_actions里选一个用replay_action(index)直接执行，不要重新用
   click/get_raw_image摸索一遍已经知道怎么做的事情
""".strip()

EXPLORATION_PROMPT_TEMPLATE = """
你正在自由探索手游【{display_name}】。
游戏特点：{exploration_goal_hint}

你的目标：尽量发现你还没见过的界面和功能。

操作规则：
{operating_rules}

本次探索最多执行 {max_steps} 步。
"""

# 接受用户直接下达的指令（而不是自由探索）时用这份模板；build_agent()/MCPServerBridge/
# 工具注册这些都是复用的，只是把"目标"从"随便探索"换成"完成指令"，并且额外开放
# known_actions/replay_action这条捷径——重复执行已知流程正是它的用武之地
INSTRUCTION_PROMPT_TEMPLATE = """
你正在操作手游【{display_name}】。
游戏特点：{exploration_goal_hint}

用户的指令：{instruction}

操作规则：
{operating_rules}
{known_actions_rule}

完成指令后调用Finish工具汇报结果；如果发现指令描述的目标不存在或做不到，也要调用Finish
说明原因，不要无限摸索。本次最多执行 {max_steps} 步。
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

    def __init__(self, bridge: MCPServerBridge, spec: McpToolSpec, *, enable_known_actions: bool = False):
        super().__init__(name=spec.name, description=spec.description or spec.name)
        self._bridge = bridge
        self._parameters = _schema_to_parameters(spec.inputSchema)
        self._enable_known_actions = enable_known_actions

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

        # MCP Server本身是无状态的、不知道当前是探索模式还是指令模式，screenshot()
        # 无条件带上known_actions/known_actions_hint。探索模式没注册replay_action
        # 工具，这两个字段留着的话提示语会让模型去调一个不存在的工具，这里剥掉
        if self.name == "screenshot" and not self._enable_known_actions and isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in ("known_actions", "known_actions_hint")}

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
        operating_rules=AGENT_OPERATING_RULES,
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )


def build_instruction_prompt(profile: dict[str, Any], instruction: str) -> str:
    return INSTRUCTION_PROMPT_TEMPLATE.format(
        display_name=profile["display_name"],
        exploration_goal_hint=profile["exploration_goal_hint"].strip(),
        instruction=instruction.strip(),
        operating_rules=AGENT_OPERATING_RULES,
        known_actions_rule=KNOWN_ACTIONS_RULE,
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )


def build_agent(
    profile_path: str | Path, *, enable_known_actions: bool = False
) -> tuple[ReActAgent, MCPServerBridge, LogBroadcaster]:
    """enable_known_actions=True时才注册replay_action工具、启用routine_llm路由。

    自由探索的目的是发现新东西，如果碰到已知节点就走known_actions捷径直接重放，
    很容易在几个互通的已知节点之间来回打转反而探索不出新内容（实测出现过"预设
    列表↔创建界面"来回重放两个已知动作、不再往前推进）。这条捷径只给"重复执行
    已知流程"的场景用（比如按用户指令执行，见build_instruction_prompt）。
    """
    profile = load_profile(profile_path)

    bridge = MCPServerBridge(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT), str(profile_path)],
        cwd=str(PROJECT_ROOT),
    )

    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    registry = ToolRegistry(circuit_breaker=circuit_breaker)
    for spec in bridge.tools:
        if not enable_known_actions and spec.name == "replay_action":
            continue
        registry.register_tool(MCPTool(bridge, spec, enable_known_actions=enable_known_actions))

    llm = HelloAgentsLLM()
    agent = ReActAgent(
        name=f"{profile['game_id']}-explorer",
        llm=llm,
        tool_registry=registry,
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )

    # 可选的第二模型：没见过的界面/探索阶段用llm（多模态），启用了known_actions捷径
    # 且命中exploration_memory已经记录过操作的界面（screenshot()返回known_actions
    # 非空）时，react_agent_vision.py的补丁会自动切到这个更便宜的纯文本模型去选
    # replay_action，不配LLM_MODEL_ID_ROUTINE或没启用known_actions就一直用llm
    routine_model_id = os.getenv("LLM_MODEL_ID_ROUTINE")
    agent.routine_llm = (
        HelloAgentsLLM(model=routine_model_id)
        if (enable_known_actions and routine_model_id)
        else None
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
        print("用法: python agent_runner.py <profile.yaml路径> [用户指令]", file=sys.stderr)
        print("      不传用户指令就是自由探索模式；传了就按指令执行，完成/做不到都会调Finish收尾", file=sys.stderr)
        sys.exit(1)

    profile_path = sys.argv[1]
    instruction = sys.argv[2] if len(sys.argv) > 2 else None
    profile = load_profile(profile_path)
    ensure_emulator_ready(profile)
    agent, bridge, broadcaster = build_agent(profile_path, enable_known_actions=bool(instruction))
    print(f"📡 实时日志面板: {broadcaster.url}")
    prompt = build_instruction_prompt(profile, instruction) if instruction else build_exploration_prompt(profile)
    try:
        agent.run(prompt)
    finally:
        bridge.close()
        broadcaster.stop()


if __name__ == "__main__":
    main()
