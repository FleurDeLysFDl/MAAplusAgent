"""离线批量合并重复节点。

exploration_memory.py的相似度算法/灰色地带图片diff兜底这些是最近才加上的，
在这之前落盘的节点数据是用旧算法判定的，会有本该是同一屏、却因为OCR噪声被
拆成好几个节点的情况（用ExplorationMemory.merge_nodes()一个个手动合并太慢）。
这个脚本用当前最新的算法重新审视所有已存节点，找出够得上合并条件的一对就
合并，重复直到没有能合并的为止。

不要在agent_runner.py还在跑的时候执行——两边会同时读写同一份exploration_logs，
状态会打架。

用法：
    python core/tools/dedupe_nodes.py <game_id>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.memory.exploration_memory import ExplorationMemory  # noqa: E402
from core.memory.image_similarity import images_look_same  # noqa: E402


def _load_image(memory: ExplorationMemory, node_id: str) -> Any:
    path = memory.nodes_dir / f"{node_id}.png"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def _find_best_merge_pair(memory: ExplorationMemory) -> tuple[str, str, float] | None:
    """全图两两比较，返回够得上合并条件、且相似度最高的一对(keep_id, remove_id, score)

    够得上合并条件：文字相似度>=SIMILARITY_THRESHOLD，或者落在灰色地带但两边
    图片diff确认是同一屏（跟maa_mcp_server.py screenshot()里在线判断用的是同一套
    标准）。keep_id取count更大的那个——访问次数更多，大概率是更早发现、
    积累的actions/description更完整的那个。
    """
    node_ids = list(memory.nodes.keys())
    best: tuple[str, str, float] | None = None
    for i, a_id in enumerate(node_ids):
        for b_id in node_ids[i + 1:]:
            score = memory.similarity_between(a_id, b_id)
            should_merge = score >= memory.SIMILARITY_THRESHOLD
            if not should_merge and memory.SIMILARITY_GRAY_ZONE <= score < memory.SIMILARITY_THRESHOLD:
                img_a = _load_image(memory, a_id)
                img_b = _load_image(memory, b_id)
                if img_a is not None and img_b is not None and images_look_same(img_a, img_b):
                    should_merge = True
            if should_merge and (best is None or score > best[2]):
                a_count = memory.nodes[a_id].get("count", 0)
                b_count = memory.nodes[b_id].get("count", 0)
                keep, remove = (a_id, b_id) if a_count >= b_count else (b_id, a_id)
                best = (keep, remove, score)
    return best


def dedupe(game_id: str) -> tuple[int, int]:
    """反复找"够得上合并条件的一对"并合并，直到图里再也找不出这样的一对为止

    每次合并后节点集合变了，必须重新扫一遍全图再找下一对——不能指望一轮扫描
    就把所有该合并的都找出来（比如A和C因为都跟B像但互相不够像，得先合并A/B，
    C再跟合并后的B比一次才可能够得上阈值）

    Returns:
        (合并次数, 合并前的节点总数)
    """
    memory = ExplorationMemory(game_id)
    before = len(memory.nodes)
    merged = 0
    while True:
        pair = _find_best_merge_pair(memory)
        if pair is None:
            break
        keep_id, remove_id, score = pair
        print(f"合并 {remove_id} -> {keep_id} (相似度={score:.3f})")
        memory.merge_nodes(keep_id, remove_id)
        merged += 1
    return merged, before


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python core/tools/dedupe_nodes.py <game_id>", file=sys.stderr)
        sys.exit(1)

    merged, before = dedupe(sys.argv[1])
    print(f"完成：合并了{merged}对节点，{before} -> {before - merged}")


if __name__ == "__main__":
    main()
