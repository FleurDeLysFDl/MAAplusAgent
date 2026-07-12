"""接受用户意图并执行：对应 软件探索Agent三大功能模块设计.md 模块1（意图执行）
+ 模块2.2场景B（指令执行模式下命中高危操作，同步阻塞询问用户）。

流程：
1. core/intent_router.py的route_intent()先定位——查已沉淀的技能库
   （core/skill_store.py），查不到就在状态图里语义搜索最接近的已知节点
2. 连接真实设备（复用core/navigate_demo.py的connect_device等），把定位结果
   真正执行出来：run_known_task就重放技能的action_sequence；
   navigate_to_state就用ExplorationMemory.path_to()导航过去（复用
   core/review_pending.py的_navigate_to）——这条路径本身不需要额外的风险
   确认，因为图里的边本来就是探索阶段已经通过RiskGate场景A校验过的已知动作，
   真正高危的候选在探索时就已经被跳过、从没成为图里的一条边
3. 只有重放技能（action_sequence可能是手工写的，没有经过场景A筛选）时才逐步
   过一遍RiskGate评估：命中高危不像自由探索那样跳过就算了，而是同步阻塞、
   在终端里直接问用户要不要继续——指令执行是用户主动发起的，用户此刻大概率
   在场，这是场景B跟场景A的核心区别
4. 已知技能库和状态图里都找不到匹配目标：进入目标导向探测（模块1.2）——按
   core/intent_router.py的rank_ocr_candidates_by_relevance给当前界面还没试过
   的候选跟intent的相关性打分，从最相关的开始试。文档原文是"LLM给候选打分，
   多步循环直到intent_satisfied()"；这里没有接LLM，"是否满足意图"这种语义
   判断没法用字符2-gram重合度硬判——与其伪造一个不可靠的自动判断，不如每次
   只试一步、把结果截图交给人看，不满足就再手动跑一次这个命令继续试下一个
   候选，多步循环退化成"人工确认后再手动重复调用"

场景B（同步询问）没有接进agent_runner.py那条真正的LLM ReAct循环——那个MCP
子进程的stdin/stdout本身就是JSON-RPC协议通道，被input()抢占会直接破坏协议，
要做到"MCP工具里也能同步问用户"需要两阶段协议改造（先"评估风险"再"确认后
执行"两次独立tool call），这个改造还没做。场景B目前只在本脚本自己的进程里
生效（重放技能、目标导向探测都在这个进程里执行，有正常的控制台，没有这个
限制）。

5. navigate_to_state成功走完一条BFS路径后，自动把这条路径固化成一条
   LearnedSkill（模块1.3"技能沉淀"）——技能名按目标节点id约定成
   f"goto_{node_id}"，同一目标第二次被路由到时不会重复固化，只是
   record_success()累加成功次数。文档原文是"被相同/相似意图触发过几次才
   固化"，这里简化成首次成功就固化：反正match_known_task有相似度门槛
   （MATCH_THRESHOLD）挡着，不相关的意图不会误命中这条技能，没有为了防止
   "一次性意图也被存下来"这个次要问题再引入一套额外的预备计数结构的必要。

用法：
    python core/execute_intent.py <profile.yaml路径> "<用户意图>"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402
from intent_router import rank_ocr_candidates_by_relevance, route_intent  # noqa: E402
from maa_mcp_server import _load_profile  # noqa: E402
from navigate_demo import _capture, _perform, _visit, connect_device  # noqa: E402
from review_pending import _navigate_to  # noqa: E402
from risk_gate import RiskGate, load_irreversible_keywords  # noqa: E402
from safety_guard import load_sensitive_keywords  # noqa: E402
from skill_store import LearnedSkill, SkillStore  # noqa: E402
from usage_log import record_usage  # noqa: E402


def _consolidate_navigate_skill(
    game_id: str, memory: ExplorationMemory, from_node_id: str, path: list[str], intent: str
) -> None:
    """模块1.3：把navigate_to_state刚走完的一条BFS路径固化成LearnedSkill，
    技能名按目标节点id约定，同一目标重复命中只累加success_count、不重复固化。
    path为空（起点就是终点，没有真的导航）时不固化——没有动作序列可存。
    """
    if not path:
        return
    sequence: list[dict[str, Any]] = []
    current = from_node_id
    for next_id in path:
        action = next(
            (a for a in memory.get_actions(current) if a.get("leads_to") == next_id), None
        )
        if action is None:
            return  # 图数据和预期对不上，放弃固化，不影响这次已经执行完的导航
        sequence.append(
            {"type": action["type"], "coords": action["coords"], "trigger_text": action.get("trigger_text")}
        )
        current = next_id

    store = SkillStore(game_id)
    skill_name = f"goto_{path[-1]}"
    if any(s.name == skill_name for s in store.list_all()):
        store.record_success(skill_name, intent)
        return
    store.add(LearnedSkill(name=skill_name, intent_examples=[intent], action_sequence=sequence))
    print(f"📌 已固化技能「{skill_name}」，下次类似意图可以直接命中已知技能，不用再走BFS导航")


def _confirm_high_risk(risk_gate: RiskGate, action: dict[str, Any]) -> bool:
    """场景B：命中高危同步阻塞询问用户，返回True表示用户放行执行。"""
    print(
        f"\n⚠️ 即将执行的操作命中风险关键词「{risk_gate.matched_keyword}」：\n"
        f"   {action['type']} {action.get('coords')} 「{action.get('trigger_text')}」\n"
    )
    return input("是否继续执行这一步？(y/N): ").strip().lower() == "y"


def _replay_skill(controller, safety_guard, risk_gate: RiskGate, skill) -> bool:
    """重放一条已知技能的动作序列，逐步过风险确认。返回False表示中途被用户拒绝。"""
    for i, action in enumerate(skill.action_sequence, start=1):
        print(f"第{i}步: {action['type']} {action.get('coords')} 「{action.get('trigger_text')}」")
        if action["type"] == "click":
            if risk_gate.assess(action.get("trigger_text")) == "high_risk":
                if not _confirm_high_risk(risk_gate, action):
                    print("已跳过，意图执行终止")
                    return False
        _perform(controller, safety_guard, action["type"], action.get("coords"))
        time.sleep(1.0)
    return True


def _probe_one_step(
    memory: ExplorationMemory, tasker, controller, safety_guard, risk_gate: RiskGate, intent: str
) -> None:
    """目标导向探测的一步：截图拿当前候选，按跟intent的相关性排序，从最相关的
    还没试过的候选开始尝试，命中风险就同步询问。只执行一步——"这一步有没有
    让意图更接近满足"需要人自己看结果判断，不在这里自动决定要不要继续。
    """
    image, ocr_texts = _capture(tasker, controller)
    node_id, _, _ = _visit(memory, image, ocr_texts)
    tried_texts = {
        a["trigger_text"]
        for a in memory.get_actions(node_id) + memory.get_ineffective_actions(node_id)
        if a.get("trigger_text")
    }

    ranked = rank_ocr_candidates_by_relevance(intent, ocr_texts, tried_texts)
    if not ranked:
        print(f"❌ 当前节点{node_id}上没有还没试过的候选了，探测中止，需要人工换个方向")
        return

    best = ranked[0]
    bx, by, bw, bh = best["box"]
    cx, cy = bx + bw // 2, by + bh // 2
    print(f"🔎 当前节点: {node_id}；按相关性排序后最靠前的候选: 「{best['text']}」")

    if risk_gate.assess(best["text"]) == "high_risk":
        action = {"type": "click", "coords": [cx, cy], "trigger_text": best["text"]}
        if not _confirm_high_risk(risk_gate, action):
            print("已跳过，探测中止")
            return

    _perform(controller, safety_guard, "click", [cx, cy])
    memory.record_action(node_id, "click", [cx, cy], trigger_text=best["text"])
    time.sleep(1.0)

    image, ocr_texts = _capture(tasker, controller)
    new_node_id, _, is_novel = _visit(memory, image, ocr_texts)
    print(f"📸 落地节点: {new_node_id}（{'新界面' if is_novel else '已知界面'}）")
    print("请自己看这一步有没有满足意图；没有的话可以再跑一次同样的命令，会自动换下一个候选")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print('用法: python execute_intent.py <profile.yaml路径> "<用户意图>"', file=sys.stderr)
        sys.exit(1)

    profile_path, intent = sys.argv[1], sys.argv[2]
    profile = _load_profile(profile_path)
    game_id = profile["game_id"]

    decision = route_intent(intent, game_id)
    print(f"🧭 定位结果: {decision['action']}")

    memory = ExplorationMemory(game_id)
    tasker, controller, safety_guard = connect_device(profile)
    risk_gate = RiskGate(load_sensitive_keywords(profile), load_irreversible_keywords(profile))

    if decision["action"] == "goal_directed_probe":
        print("🔍 已知技能库和状态图里都没有找到匹配目标，进入目标导向探测（模块1.2）")
        _probe_one_step(memory, tasker, controller, safety_guard, risk_gate, intent)
        record_usage(game_id, "intent_execute", intent, "probed_one_step", extra={"route": "goal_directed_probe"})
        return

    if decision["action"] == "run_known_task":
        skill = decision["skill"]
        print(f"📖 命中已知技能「{skill.name}」（已成功{skill.success_count}次），重放其操作序列")
        if not _replay_skill(controller, safety_guard, risk_gate, skill):
            record_usage(
                game_id, "intent_execute", intent, "rejected_by_user",
                extra={"route": "run_known_task", "skill": skill.name},
            )
            return
        SkillStore(game_id).record_success(skill.name, intent)
        print("✅ 技能重放完成")
        record_usage(
            game_id, "intent_execute", intent, "success",
            extra={"route": "run_known_task", "skill": skill.name},
        )
        return

    node_id, score = decision["node_id"], decision["score"]
    print(f"🎯 状态图里最接近的已知节点: {node_id}（相似度{score:.2f}）")

    image, ocr_texts = _capture(tasker, controller)
    current_id, _, _ = _visit(memory, image, ocr_texts)
    path = memory.path_to(current_id, node_id)

    landed = _navigate_to(memory, tasker, controller, safety_guard, node_id)
    if landed is None:
        print("❌ 没能导航到目标节点")
        record_usage(
            game_id, "intent_execute", intent, "failed: no path",
            extra={"route": "navigate_to_state", "node_id": node_id},
        )
        sys.exit(1)
    print(f"✅ 已到达 {node_id}——意图执行的\"定位\"部分完成，到达后具体怎么完成意图仍需人工/后续LLM接管")
    record_usage(
        game_id, "intent_execute", intent, "success",
        extra={"route": "navigate_to_state", "node_id": node_id},
    )

    if path:
        _consolidate_navigate_skill(game_id, memory, current_id, path, intent)


if __name__ == "__main__":
    main()
