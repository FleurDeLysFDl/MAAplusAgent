"""core/risk_gate.py的单元测试：风险评估 + 待确认队列的持久化。"""
from __future__ import annotations

from risk_gate import DEFAULT_IRREVERSIBLE_KEYWORDS, PendingConfirmationQueue, RiskGate


class TestRiskGateAssess:
    def test_safe_text(self):
        gate = RiskGate(["充值", "支付"], DEFAULT_IRREVERSIBLE_KEYWORDS)
        assert gate.assess("前往商城") == "safe"

    def test_sensitive_keyword_high_risk(self):
        gate = RiskGate(["充值", "支付"], DEFAULT_IRREVERSIBLE_KEYWORDS)
        assert gate.assess("立即充值") == "high_risk"
        assert gate.matched_keyword == "充值"

    def test_irreversible_keyword_high_risk(self):
        gate = RiskGate([], DEFAULT_IRREVERSIBLE_KEYWORDS)
        assert gate.assess("清空购物车") == "high_risk"
        assert gate.matched_keyword == "清空"

    def test_common_confirm_words_not_flagged(self):
        """"确定""确认""提交"故意不在默认不可逆词表里——太通用，日常关闭弹窗
        都会用到，收进来只会疯狂误报。"""
        gate = RiskGate([], DEFAULT_IRREVERSIBLE_KEYWORDS)
        assert gate.assess("确定") == "safe"
        assert gate.assess("确认") == "safe"
        assert gate.assess("提交") == "safe"

    def test_none_text_is_safe(self):
        """click_on_image这类没有OCR文字的图标点击，没有可判断的文字依据。"""
        gate = RiskGate(["充值"], DEFAULT_IRREVERSIBLE_KEYWORDS)
        assert gate.assess(None) == "safe"

    def test_matched_keyword_resets_between_calls(self):
        gate = RiskGate(["充值"], [])
        gate.assess("立即充值")
        assert gate.matched_keyword == "充值"
        gate.assess("安全的文字")
        assert gate.matched_keyword is None


class TestPendingConfirmationQueue:
    def test_add_and_list(self, tmp_path):
        queue = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        queue.add("g-1", "click", [1, 2], "充值按钮", "充值")
        items = queue.list_all()
        assert len(items) == 1
        assert items[0]["node_id"] == "g-1"
        assert items[0]["matched_keyword"] == "充值"

    def test_persists_across_instances(self, tmp_path):
        queue = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        queue.add("g-1", "click", [1, 2], "充值按钮", "充值")

        reloaded = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        assert len(reloaded.list_all()) == 1

    def test_remove_by_index(self, tmp_path):
        queue = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        queue.add("g-1", "click", [1, 2], "第一条", "充值")
        queue.add("g-2", "click", [3, 4], "第二条", "支付")

        removed = queue.remove(0)
        assert removed["trigger_text"] == "第一条"
        remaining = queue.list_all()
        assert len(remaining) == 1
        assert remaining[0]["trigger_text"] == "第二条"

    def test_clear_empties_queue(self, tmp_path):
        queue = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        queue.add("g-1", "click", [1, 2], "a", "充值")
        queue.add("g-2", "click", [1, 2], "b", "支付")

        removed_count = queue.clear()
        assert removed_count == 2
        assert queue.list_all() == []

    def test_missing_file_starts_empty(self, tmp_path):
        queue = PendingConfirmationQueue("g", storage_dir=str(tmp_path))
        assert queue.list_all() == []
