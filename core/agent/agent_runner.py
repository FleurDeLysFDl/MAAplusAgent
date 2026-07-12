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
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from mcp.types import Tool as McpToolSpec

from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.skills import SkillLoader
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.builtin.skill_tool import SkillTool
from hello_agents.tools.circuit_breaker import CircuitBreaker
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.response import ToolResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.device.emulator import ensure_emulator_ready  # noqa: E402
from core.agent.log_broadcaster import LogBroadcaster, attach_broadcaster  # noqa: E402
from core.agent.react_agent_vision import install_vision_support  # noqa: E402
from core.memory.usage_log import record_usage  # noqa: E402

install_vision_support()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_SCRIPT = PROJECT_ROOT / "core" / "device" / "maa_mcp_server.py"

DEFAULT_MAX_STEPS = 30
DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8765

# core/agent/idle_scheduler.py（软件探索Agent三大功能模块设计.md模块0.1）靠这个文件
# 的mtime判断"用户最近有没有发过指令"——用户带指令跑agent_runner.py就是最直接
# 的"用户在场"信号，touch一下这个文件，调度器只读不写，不需要额外的IPC机制
USER_ACTIVITY_MARKER = PROJECT_ROOT / "memory" / "last_user_activity.txt"


def record_user_activity() -> None:
    USER_ACTIVITY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    USER_ACTIVITY_MARKER.write_text(str(time.time()), encoding="utf-8")

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

# known_actions/replay_action这条捷径两种模式现在都给：重启会话时探索记忆里已经记住的
# 节点/操作不会丢，靠这条捷径跳过对已知界面的重复视觉摸索，只把力气花在真正没去过的地方。
# 唯一的风险是探索模式下模型可能在几个互通的已知节点之间来回打转、不再往前推进（实测出现
# 过"预设列表↔创建界面"来回重放两个已知动作的情况）——这个风险由ExplorationStallGuard
# 兜底：探索模式下连续多次screenshot()都没发现新节点，就临时收回known_actions，逼它换成
# click/get_raw_image手动摸索；一旦摸出新节点，捷径自动恢复。指令执行模式不装这个guard，
# 重复执行已知流程本来就是它的正常用途，没有"必须发现新东西"的压力，不需要防打转。
KNOWN_ACTIONS_RULE = """
7. 如果screenshot()返回结果里known_actions不为空，说明这个界面之前来过、已经有验证过
   坐标的操作，优先从known_actions里选一个用replay_action(index)直接执行，不要重新用
   click/get_raw_image摸索一遍已经知道怎么做的事情
""".strip()

EXPLORATION_PROMPT_TEMPLATE = """
你正在自由探索手游【{display_name}】。
游戏特点：{exploration_goal_hint}

你的目标：尽量发现你还没见过的界面和功能。
{min_new_nodes_rule}
操作规则：
{operating_rules}
{known_actions_rule}

本次探索最多执行 {max_steps} 步。
"""

# 接受用户直接下达的指令（而不是自由探索）时用这份模板；build_agent()/MCPServerBridge/
# 工具注册这些都是复用的，只是把"目标"从"随便探索"换成"完成指令"
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


class ExplorationStallGuard:
    """自由探索模式下known_actions捷径的动态开关。

    只在build_agent(instruction=None)的自由探索会话里创建；指令执行模式不装这个guard，
    known_actions永远放行——重复执行已知流程正是指令模式的正常用途，没有"必须发现新
    东西"的压力，不存在打转风险。

    连续STALL_THRESHOLD次screenshot()都没发现新节点（is_novel=False），说明这段时间
    模型可能在几个互通的已知节点之间来回打转（实测出现过"预设列表↔创建界面"来回重放
    两个已知动作、不再往前推进的情况），这时暂时收回known_actions/known_actions_hint，
    逼模型换成click/get_raw_image手动摸索；一旦摸出一个新节点（is_novel=True），
    stall计数清零，捷径立刻恢复。
    """

    STALL_THRESHOLD = 4

    def __init__(self) -> None:
        self._stall_streak = 0

    @property
    def known_actions_enabled(self) -> bool:
        return self._stall_streak < self.STALL_THRESHOLD

    def record(self, is_novel: bool) -> None:
        self._stall_streak = 0 if is_novel else self._stall_streak + 1


class MCPTool(Tool):
    """把MAA-MCP-Server的一个远程工具包装成hello_agents的Tool"""

    def __init__(
        self,
        bridge: MCPServerBridge,
        spec: McpToolSpec,
        *,
        stall_guard: ExplorationStallGuard | None = None,
    ):
        super().__init__(name=spec.name, description=spec.description or spec.name)
        self._bridge = bridge
        self._parameters = _schema_to_parameters(spec.inputSchema)
        self._stall_guard = stall_guard

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

        # stall_guard is None（指令执行模式）：known_actions/known_actions_hint原样放行。
        # stall_guard存在（自由探索模式）：先用这次screenshot的is_novel喂给guard更新
        # 打转计数，guard判定"最近连续多次都没发现新节点"时收回这两个字段
        if self.name == "screenshot" and isinstance(data, dict) and self._stall_guard is not None:
            self._stall_guard.record(bool(data.get("is_novel")))
            if not self._stall_guard.known_actions_enabled:
                data = {k: v for k, v in data.items() if k not in ("known_actions", "known_actions_hint")}

        text = data.get("sensitive_warning") if isinstance(data, dict) else None
        if not text:
            text = json.dumps(data, ensure_ascii=False)
        return ToolResponse.success(text=text, data=data if isinstance(data, dict) else {"result": data})


def load_profile(profile_path: str | Path) -> dict[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_exploration_prompt(profile: dict[str, Any]) -> str:
    min_new_nodes = profile.get("min_new_nodes", 0)
    # 光靠Prompt里说"尽量发现"管不住模型——实测发现个位数新界面后就会主动调Finish喊停。
    # min_new_nodes>0时这里只是提前把硬性目标亮出来，真正的强制力在
    # react_agent_vision.py._patched_run_impl里拦Finish工具调用那部分
    min_new_nodes_rule = (
        f"本次探索目标：至少发现{min_new_nodes}个新界面（不是重复访问到的）才能结束，"
        "不要找到几个新界面就急着调用Finish收尾。\n"
        if min_new_nodes > 0
        else ""
    )
    return EXPLORATION_PROMPT_TEMPLATE.format(
        display_name=profile["display_name"],
        exploration_goal_hint=profile["exploration_goal_hint"].strip(),
        min_new_nodes_rule=min_new_nodes_rule,
        operating_rules=AGENT_OPERATING_RULES,
        known_actions_rule=KNOWN_ACTIONS_RULE,
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
    profile_path: str | Path, *, is_exploration: bool = True
) -> tuple[ReActAgent, MCPServerBridge, LogBroadcaster]:
    """is_exploration=True是自由探索会话（没有用户指令），False是按指令执行。

    两种模式现在都注册replay_action、都放行known_actions/routine_llm捷径——重启
    会话时探索记忆里已经记住的节点/操作不会丢，没道理让模型每次都从视觉摸索重新
    走一遍已知界面。区别只在于自由探索模式额外装了ExplorationStallGuard防打转
    （见其docstring），指令执行模式没有"必须发现新东西"的压力，不需要这个guard，
    也不受min_new_nodes硬性目标约束——完成指令就该立刻Finish。
    """
    profile = load_profile(profile_path)

    # --auto-navigate只在自由探索模式开：已知节点摸到没有候选目标了(has_frontier=False)
    # 就沿已知有效边自动跳到最近一个还有候选目标的节点，见maa_mcp_server.py里screenshot()
    # 的说明。指令执行模式要精确控制每一步，不该被自动带去别的地方
    bridge = MCPServerBridge(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT), str(profile_path)] + (["--auto-navigate"] if is_exploration else []),
        cwd=str(PROJECT_ROOT),
    )

    stall_guard = ExplorationStallGuard() if is_exploration else None
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    registry = ToolRegistry(circuit_breaker=circuit_breaker)
    for spec in bridge.tools:
        registry.register_tool(MCPTool(bridge, spec, stall_guard=stall_guard))

    # skills/<name>/SKILL.md：游戏无关的领域知识（比如通用手游UI导航套路），
    # 按需加载，不占system prompt的常驻token。探索/指令两种模式都注册——
    # 跟known_actions那种"重复已知流程"的捷径不一样，这里存的是可迁移的通用
    # 经验，不是某个具体节点的坐标，不会导致"探索目的被绕过"的问题
    skill_loader = SkillLoader(skills_dir=PROJECT_ROOT / "skills")
    registry.register_tool(SkillTool(skill_loader=skill_loader))

    llm = HelloAgentsLLM()
    agent = ReActAgent(
        name=f"{profile['game_id']}-explorer",
        llm=llm,
        tool_registry=registry,
        max_steps=profile.get("max_steps", DEFAULT_MAX_STEPS),
    )

    # 可选的第二模型：没见过的界面用llm（多模态），命中exploration_memory已经记录过
    # 操作的界面（screenshot()返回known_actions非空）时，react_agent_vision.py的补丁
    # 会自动切到这个更便宜的纯文本模型去选replay_action，不配LLM_MODEL_ID_ROUTINE
    # 就一直用llm。ExplorationStallGuard收回known_actions时这个字段自然也是空的，
    # 会自动切回多模态，不需要在这里单独处理
    routine_model_id = os.getenv("LLM_MODEL_ID_ROUTINE")
    agent.routine_llm = HelloAgentsLLM(model=routine_model_id) if routine_model_id else None

    # 只在自由探索模式下生效——指令执行模式完成指令就该立刻Finish，不该被
    # "至少发现N个新界面"这条规则拖着继续摸索
    agent.min_new_nodes = profile.get("min_new_nodes", 0) if is_exploration else 0

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
        print("用法: python core/agent/agent_runner.py <profile.yaml路径> [用户指令]", file=sys.stderr)
        print("      不传用户指令就是自由探索模式；传了就按指令执行，完成/做不到都会调Finish收尾", file=sys.stderr)
        sys.exit(1)

    profile_path = sys.argv[1]
    instruction = sys.argv[2] if len(sys.argv) > 2 else None
    if instruction:
        record_user_activity()
    profile = load_profile(profile_path)
    ensure_emulator_ready(profile)
    agent, bridge, broadcaster = build_agent(profile_path, is_exploration=not instruction)
    print(f"📡 实时日志面板: {broadcaster.url}")
    prompt = build_instruction_prompt(profile, instruction) if instruction else build_exploration_prompt(profile)
    game_id = profile["game_id"]
    mode = "instruction" if instruction else "explore"
    try:
        final_answer = agent.run(prompt)
        record_usage(game_id, mode, instruction, final_answer)
    except Exception as e:
        record_usage(game_id, mode, instruction, f"error: {e}")
        raise
    finally:
        bridge.close()
        broadcaster.stop()


if __name__ == "__main__":
    main()
