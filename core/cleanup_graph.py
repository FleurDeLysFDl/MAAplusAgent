"""离线整理探索记忆里的节点/操作数据：清掉悬空边、合并重复动作条目，可选删除
孤立节点。

跟dedupe_nodes.py处理的是不同的问题——那边合并"应该是同一屏却被OCR噪声判成
两个节点"的重复节点，这里处理的是同一份数据内部的一致性问题（悬空边、重复
动作条目），默认也不删除/合并任何节点本身。

孤立节点（既无入边也无出边）删不删需要人工判断——可能是OCR噪声偶然判成的
垃圾数据，也可能是还没找到路径进去的真实存在的界面，所以默认不删，只有显式
传--delete-isolated才会列出来、确认后删除。

不要在agent_runner.py还在跑的时候执行——两边会同时读写同一份exploration_logs，
状态会打架。

用法：
    python core/cleanup_graph.py <game_id> [--delete-isolated]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python cleanup_graph.py <game_id> [--delete-isolated]", file=sys.stderr)
        sys.exit(1)

    game_id = sys.argv[1]
    delete_isolated = "--delete-isolated" in sys.argv[2:]

    memory = ExplorationMemory(game_id)
    stats = memory.cleanup()
    print(
        f"整理完成：删除悬空边 {stats['dangling_edges_removed']} 条，"
        f"合并重复动作 {stats['duplicate_actions_merged']} 条"
    )

    if not delete_isolated:
        isolated = memory.isolated_nodes()
        if isolated:
            print(
                f"另外发现 {len(isolated)} 个孤立节点（既无入边也无出边）：{isolated}\n"
                "可能是OCR噪声误判的垃圾数据，也可能是还没找到路径进去的真实界面，"
                "默认不删除；确认要删的话加 --delete-isolated 重新运行"
            )
        return

    isolated = memory.isolated_nodes()
    if not isolated:
        print("没有孤立节点，不需要删除")
        return

    print(f"即将删除以下 {len(isolated)} 个孤立节点：")
    for node_id in isolated:
        desc = memory.nodes[node_id].get("description", "")
        print(f"  {node_id}  {desc}")
    choice = input("确认删除？(y/N): ").strip().lower()
    if choice != "y":
        print("已取消，未删除")
        return

    for node_id in isolated:
        memory.delete_node(node_id)
    print(f"已删除 {len(isolated)} 个孤立节点")

    # 删除节点可能让别的节点上残留指向它们的悬空边，顺手再清一遍
    stats2 = memory.cleanup()
    if stats2["dangling_edges_removed"]:
        print(f"清理删除孤立节点产生的悬空边 {stats2['dangling_edges_removed']} 条")


if __name__ == "__main__":
    main()
