"""core/memory/progress_estimate.py的单元测试：frontier_progress读图数据，
discovery_rate读trace日志，两个指标要能反映真实情况（尤其是"卡住"和"探完"
这两种容易混淆的场景）。
"""
from __future__ import annotations

import json

from core.memory.exploration_memory import ExplorationMemory
from core.memory.progress_estimate import discovery_rate, estimate_progress, frontier_progress


def ocr_item(text: str, box: list[int]) -> dict:
    return {"text": text, "box": box}


def write_trace(path, novel_flags: list[bool]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for flag in novel_flags:
            line = {
                "event": "tool_result",
                "payload": {
                    "tool_name": "screenshot",
                    "result": json.dumps({"is_novel": flag}),
                },
            }
            f.write(json.dumps(line) + "\n")


class TestFrontierProgress:
    def test_empty_memory_zero(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        assert frontier_progress(memory) == 0.0

    def test_all_tried_gives_full_progress(self, tmp_path):
        """frontier_progress汇总的是"所有已知节点"的已试/未试候选，不只是起点——
        终点节点自己的标签文字，如果从没有从终点出发再点过它，也算一个未试候选。
        要让整体progress到1.0，两个节点各自的候选都得被"从该节点出发的动作"覆盖到。
        """
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit([ocr_item("按钮A", [0, 0, 1, 1])])
        memory.record_action(node_a, "click", [0, 0], trigger_text="按钮A")
        node_b, _, _ = memory.visit([ocr_item("按钮B", [0, 0, 1, 1])])
        memory.record_action(node_b, "click", [0, 0], trigger_text="按钮B")
        memory.visit([ocr_item("按钮A", [0, 0, 1, 1])])  # 回到A，解决B的挂起动作

        assert frontier_progress(memory) == 1.0

    def test_partial_progress_between_zero_and_one(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_id, _, _ = memory.visit(
            [ocr_item("已试按钮", [0, 0, 1, 1]), ocr_item("未试按钮", [1, 1, 1, 1])]
        )
        memory.record_action(node_id, "click", [0, 0], trigger_text="已试按钮")
        memory.visit([ocr_item("下一屏", [0, 0, 1, 1])])

        progress = frontier_progress(memory)
        assert 0.0 < progress < 1.0


class TestDiscoveryRate:
    def test_all_novel_recent(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        write_trace(trace_path, [True, True, True])
        assert discovery_rate(trace_path, window=20) == 1.0

    def test_none_novel_recent(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        write_trace(trace_path, [False, False, False])
        assert discovery_rate(trace_path, window=20) == 0.0

    def test_only_considers_recent_window(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        write_trace(trace_path, [True, True, True] + [False] * 20)
        assert discovery_rate(trace_path, window=20) == 0.0

    def test_missing_file_returns_one(self, tmp_path):
        assert discovery_rate(tmp_path / "不存在.jsonl") == 1.0

    def test_empty_trace_returns_one(self, tmp_path):
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text("", encoding="utf-8")
        assert discovery_rate(trace_path) == 1.0


class TestEstimateProgress:
    def test_stuck_scenario_is_not_marked_complete(self, tmp_path):
        """回归用例：2026-07-11那次wuqimitu真实卡住的会话——discovery_rate=0，
        但frontier_progress远没到0.9，verdict不应该是"接近探完"。"""
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_id, _, _ = memory.visit(
            [ocr_item(f"未试按钮{i}", [i, i, 1, 1]) for i in range(20)]
        )
        memory.record_action(node_id, "click", [0, 0], trigger_text="未试按钮0")
        memory.visit([ocr_item("下一屏", [0, 0, 1, 1])])

        trace_path = tmp_path / "trace.jsonl"
        write_trace(trace_path, [False] * 20)

        result = estimate_progress(memory, trace_path)
        assert result["verdict"] == "探索中"
        assert result["recent_discovery_rate"] == 0.0
        assert result["frontier_progress"] < 0.9

    def test_thoroughly_explored_marked_complete(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit([ocr_item("按钮A", [0, 0, 1, 1])])
        memory.record_action(node_a, "click", [0, 0], trigger_text="按钮A")
        node_b, _, _ = memory.visit([ocr_item("按钮B", [0, 0, 1, 1])])
        memory.record_action(node_b, "click", [0, 0], trigger_text="按钮B")
        memory.visit([ocr_item("按钮A", [0, 0, 1, 1])])

        trace_path = tmp_path / "trace.jsonl"
        write_trace(trace_path, [False] * 20)

        result = estimate_progress(memory, trace_path)
        assert result["frontier_progress"] == 1.0
        assert result["verdict"] == "接近探完"
