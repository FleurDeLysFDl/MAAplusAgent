"""core/usage_log.py的单元测试：不同app（game_id）的使用记录互不干扰。"""
from __future__ import annotations

from usage_log import read_usage_log, record_usage, summarize_usage


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
