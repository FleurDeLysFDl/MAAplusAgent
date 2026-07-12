"""场景A的收尾：人工审阅core/risk_gate.py积累的待确认队列，选择性放行执行。

对应 软件探索Agent三大功能模块设计.md 模块2.2场景A最后一句"等用户下次上线，
把待确认队列展示出来，用户可以选择性放行"。复用core/navigate_demo.py已经写好
的截图/OCR/节点匹配/动作执行这几块——放行的候选如果当前不在记录时所在的那个
节点上，先用ExplorationMemory.path_to()导航过去，再真的执行这次点击；执行完
（或者用户决定丢弃不管）都要把这一条从队列里摘掉，不然下次审阅还会重复看到。

用法：
    python core/review_pending.py <profile.yaml路径>
交互：
    输入序号        放行执行这一条（自动导航到对应节点后再点）
    d<序号>         丢弃这一条，不执行
    dall            清空整个队列，全部丢弃
    直接回车/q      不处理，退出
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402
from maa_mcp_server import _load_profile  # noqa: E402
from navigate_demo import _capture, _perform, _visit, connect_device  # noqa: E402
from risk_gate import PendingConfirmationQueue  # noqa: E402


def _print_queue(items: list[dict]) -> None:
    for i, item in enumerate(items):
        print(
            f"[{i}] 节点{item['node_id']} · {item['type']} {item['coords']} · "
            f"文字「{item['trigger_text']}」· 命中「{item['matched_keyword']}」· {item['ts']}"
        )


def _navigate_to(memory: ExplorationMemory, tasker, controller, safety_guard, target_id: str) -> str | None:
    """导航到target_id，返回实际落地的节点id；中途对不上就提前返回None。"""
    image, ocr_texts = _capture(tasker, controller)
    current, _, _ = _visit(memory, image, ocr_texts)
    if current == target_id:
        return current

    path = memory.path_to(current, target_id)
    if path is None:
        print(f"❌ 已知图里没有从 {current} 到 {target_id} 的路径，无法自动导航过去")
        return None

    print(f"🧭 导航路径（{len(path)}跳）: {current} -> {' -> '.join(path)}")
    for next_id in path:
        action = next(
            (a for a in memory.get_actions(current) if a.get("leads_to") == next_id),
            None,
        )
        if action is None:
            print(f"❌ {current}上没有找到通向{next_id}的已知动作，导航中止")
            return None
        _perform(controller, safety_guard, action["type"], action["coords"])
        memory.record_action(
            current, action["type"], action["coords"],
            trigger_text=action.get("trigger_text"),
            used_vision=action.get("used_vision", False),
        )
        time.sleep(1.0)
        image, ocr_texts = _capture(tasker, controller)
        landed_id, _, _ = _visit(memory, image, ocr_texts)
        if landed_id != next_id:
            print(f"⚠️ 落地节点是{landed_id}，跟预期的{next_id}不符，导航中止")
            return None
        current = landed_id
    return current


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python review_pending.py <profile.yaml路径>", file=sys.stderr)
        sys.exit(1)

    profile = _load_profile(sys.argv[1])
    game_id = profile["game_id"]
    queue = PendingConfirmationQueue(game_id)

    if not queue.list_all():
        print("待确认队列是空的，没有需要审阅的高危候选")
        return

    print(f"共{len(queue.list_all())}条待确认：")
    _print_queue(queue.list_all())

    choice = input(
        "\n输入序号放行执行；d<序号>丢弃某一条；dall清空全部；直接回车/q不处理退出: "
    ).strip()

    if not choice or choice.lower() == "q":
        print("已保留，不做处理")
        return

    if choice.lower() == "dall":
        n = queue.clear()
        print(f"已清空全部{n}条，均未执行")
        return

    if choice.lower().startswith("d"):
        idx = int(choice[1:])
        item = queue.remove(idx)
        print(f"已丢弃: {item}，未执行")
        return

    idx = int(choice)
    item = queue.list_all()[idx]
    print(
        f"\n即将真正执行: 节点{item['node_id']} · {item['type']} {item['coords']} · "
        f"「{item['trigger_text']}」（命中「{item['matched_keyword']}」）"
    )

    memory = ExplorationMemory(game_id)
    tasker, controller, safety_guard = connect_device(profile)

    landed = _navigate_to(memory, tasker, controller, safety_guard, item["node_id"])
    if landed is None:
        print("❌ 没能导航到目标节点，放弃执行，这一条继续留在待确认队列里")
        return

    _perform(controller, safety_guard, item["type"], item["coords"])
    memory.record_action(
        landed, item["type"], item["coords"], trigger_text=item.get("trigger_text")
    )
    queue.remove(idx)
    print("✅ 已执行，并从待确认队列移除")


if __name__ == "__main__":
    main()
