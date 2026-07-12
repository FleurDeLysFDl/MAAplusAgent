"""core/usage_log.py的单元测试：不同app（game_id）的使用记录互不干扰。"""
from __future__ import annotations

from usage_log import build_usage_context, read_usage_log, record_usage, summarize_usage


class TestRecordAndReadUsage:
    def test_records_persist_in_order(self, tmp_path):
        record_usage("gameA", "explore", None, "success", storage_dir=str(tmp_path))
        record_usage("gameA", "instruction", "签到", "success", storage_dir=str(tmp_path))

        entries = read_usage_log("gameA", storage_dir=str(tmp_path))
        assert len(entries) == 2
        assert entries[0]["mode"] == "explore"
        assert entries[1]["input"] == "签到"

    def test_different_apps_do_not_interfere(self, tmp_path):
        record_usage("gameA", "explore", None, "success", storage_dir=str(tmp_path))
        record_usage("gameB", "instruction", "去设置", "success", storage_dir=str(tmp_path))

        assert len(read_usage_log("gameA", storage_dir=str(tmp_path))) == 1
        assert len(read_usage_log("gameB", storage_dir=str(tmp_path))) == 1

    def test_extra_fields_are_included(self, tmp_path):
        record_usage(
            "gameA", "intent_execute", "去商城", "success",
            storage_dir=str(tmp_path), extra={"route": "navigate_to_state", "node_id": "gameA-4"},
        )
        entry = read_usage_log("gameA", storage_dir=str(tmp_path))[0]
        assert entry["route"] == "navigate_to_state"
        assert entry["node_id"] == "gameA-4"

    def test_missing_log_returns_empty(self, tmp_path):
        assert read_usage_log("从未使用过", storage_dir=str(tmp_path)) == []

    def test_corrupted_line_is_skipped(self, tmp_path):
        record_usage("gameA", "explore", None, "success", storage_dir=str(tmp_path))
        path = tmp_path / "gameA" / "usage_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write("这不是合法json\n")
        record_usage("gameA", "explore", None, "success2", storage_dir=str(tmp_path))

        entries = read_usage_log("gameA", storage_dir=str(tmp_path))
        assert len(entries) == 2  # 损坏的那一行被跳过，不影响前后正常记录


class TestSummarizeUsage:
    def test_empty_summary(self, tmp_path):
        summary = summarize_usage("从未使用过", storage_dir=str(tmp_path))
        assert summary == {"total": 0, "by_mode": {}, "last": None}

    def test_groups_by_mode_and_tracks_last(self, tmp_path):
        record_usage("gameA", "explore", None, "success", storage_dir=str(tmp_path))
        record_usage("gameA", "explore", None, "timeout", storage_dir=str(tmp_path))
        record_usage("gameA", "instruction", "签到", "success", storage_dir=str(tmp_path))

        summary = summarize_usage("gameA", storage_dir=str(tmp_path))
        assert summary["total"] == 3
        assert summary["by_mode"] == {"explore": 2, "instruction": 1}
        assert summary["last"]["input"] == "签到"


class TestBuildUsageContext:
    def test_empty_history(self, tmp_path):
        context = build_usage_context("从未使用过", storage_dir=str(tmp_path))
        assert "还没有使用记录" in context

    def test_few_entries_all_kept_in_full_detail(self, tmp_path):
        """条数没超过recent_n时，不应该出现"更早的...次使用"这种压缩摘要——
        全部都在"最近"范围内，没有需要压缩的部分。"""
        for i in range(3):
            record_usage("gameA", "explore", None, f"success{i}", storage_dir=str(tmp_path))

        context = build_usage_context("gameA", recent_n=10, storage_dir=str(tmp_path))
        assert "更早的" not in context
        assert "最近3次使用" in context

    def test_old_entries_compressed_not_listed_individually(self, tmp_path):
        """回归核心行为：条数超过recent_n后，长度不能随记录数无限增长——早期
        记录只出现在聚合统计里，不会逐条罗列。"""
        for i in range(20):
            record_usage("gameA", "explore", None, "success", storage_dir=str(tmp_path))

        context = build_usage_context("gameA", recent_n=5, storage_dir=str(tmp_path))
        assert "更早的15次使用" in context
        assert "最近5次使用" in context
        # 压缩后的文本长度不应该跟着entries数量线性增长：20条里只有5条被逐条列出
        assert context.count("success") <= 5 + 5  # 5条细节 + by_result摘要里最多再出现一次量级的引用

    def test_compression_bounds_size_regardless_of_history_length(self, tmp_path):
        """这是"上下文压缩"这个需求的核心验证：哪怕历史记录涨到200条，压缩后
        文本长度也不应该跟着线性暴涨——多出来的历史应该都被计入同一组聚合统计，
        而不是每条都变成新的一行。"""
        for i in range(200):
            record_usage("gameA", "instruction", f"指令{i}", "success", storage_dir=str(tmp_path))

        context_200 = build_usage_context("gameA", recent_n=10, storage_dir=str(tmp_path))

        # 再加200条，长度增量应该很小（只有汇总数字变了，不是逐条累加）
        for i in range(200, 400):
            record_usage("gameA", "instruction", f"指令{i}", "success", storage_dir=str(tmp_path))
        context_400 = build_usage_context("gameA", recent_n=10, storage_dir=str(tmp_path))

        assert len(context_400) < len(context_200) * 1.5

    def test_different_result_prefixes_grouped_by_category(self, tmp_path):
        for i in range(12):
            record_usage("gameA", "explore", None, f"error: 具体报错{i}各不相同", storage_dir=str(tmp_path))

        context = build_usage_context("gameA", recent_n=2, storage_dir=str(tmp_path))
        # 10条落在"更早"范围，都是error类，应该被合并成同一个分类计数，而不是
        # 10种不同的报错文本各算一类
        assert "'error': 10" in context
