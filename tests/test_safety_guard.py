"""core/safety_guard.py的单元测试：坐标白名单校验、禁止点击区域、敏感词拦截。"""
from __future__ import annotations

import pytest
import yaml

from safety_guard import (
    GLOBAL_SENSITIVE_KEYWORDS,
    CoordinateRejected,
    SafetyGuard,
    load_click_forbidden_keywords,
    load_sensitive_keywords,
)


def ocr(text: str, box: list[int]) -> dict:
    return {"text": text, "box": box}


class TestLoadKeywords:
    def test_sensitive_keywords_merges_global_and_extra(self, tmp_path):
        yaml_path = tmp_path / "sensitive.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"extra": ["首充"], "click_forbidden": ["追踪"]}, allow_unicode=True),
            encoding="utf-8",
        )
        keywords = load_sensitive_keywords({"sensitive_keywords_path": str(yaml_path)})
        assert set(GLOBAL_SENSITIVE_KEYWORDS).issubset(set(keywords))
        assert "首充" in keywords

    def test_click_forbidden_keywords(self, tmp_path):
        yaml_path = tmp_path / "sensitive.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"click_forbidden": ["追踪"]}, allow_unicode=True), encoding="utf-8"
        )
        assert load_click_forbidden_keywords({"sensitive_keywords_path": str(yaml_path)}) == ["追踪"]

    def test_missing_path_returns_defaults_only(self):
        assert load_sensitive_keywords({}) == GLOBAL_SENSITIVE_KEYWORDS
        assert load_click_forbidden_keywords({}) == []

    def test_nonexistent_file_falls_back_gracefully(self):
        keywords = load_sensitive_keywords({"sensitive_keywords_path": "不存在的文件.yaml"})
        assert keywords == GLOBAL_SENSITIVE_KEYWORDS


class TestSafetyGuardClickValidation:
    def test_no_screenshot_yet_rejects(self):
        guard = SafetyGuard([])
        with pytest.raises(CoordinateRejected):
            guard.validate_click(10, 10)

    def test_click_inside_box_passes(self):
        guard = SafetyGuard([])
        guard.register_screenshot([ocr("按钮", [0, 0, 20, 20])])
        guard.validate_click(10, 10)  # 不抛异常就是通过

    def test_click_outside_all_boxes_rejects(self):
        guard = SafetyGuard([])
        guard.register_screenshot([ocr("按钮", [0, 0, 20, 20])])
        with pytest.raises(CoordinateRejected):
            guard.validate_click(500, 500)

    def test_click_near_forbidden_keyword_rejects(self):
        guard = SafetyGuard([], click_forbidden_keywords=["追踪"])
        guard.register_screenshot([ocr("追踪任务", [100, 100, 20, 20])])
        with pytest.raises(CoordinateRejected):
            guard.validate_click(110, 110)

    def test_click_on_image_skips_box_whitelist_but_checks_forbidden(self):
        guard = SafetyGuard([], click_forbidden_keywords=["追踪"])
        guard.register_screenshot([ocr("追踪任务", [100, 100, 20, 20])])
        # 不在任何OCR候选框内，但click_on_image不校验候选框，只要不在禁止区域就该放行
        guard.validate_click_on_image(500, 500)
        with pytest.raises(CoordinateRejected):
            guard.validate_click_on_image(110, 110)


class TestSafetyGuardSwipeValidation:
    def test_both_endpoints_must_be_in_boxes(self):
        guard = SafetyGuard([])
        guard.register_screenshot([ocr("起点", [0, 0, 20, 20]), ocr("终点", [100, 100, 20, 20])])
        guard.validate_swipe(10, 10, 110, 110)

    def test_endpoint_outside_boxes_rejects(self):
        guard = SafetyGuard([])
        guard.register_screenshot([ocr("起点", [0, 0, 20, 20])])
        with pytest.raises(CoordinateRejected):
            guard.validate_swipe(10, 10, 500, 500)


class TestSafetyGuardSensitiveKeywords:
    def test_register_screenshot_returns_hits(self):
        guard = SafetyGuard(["充值"])
        hits = guard.register_screenshot([ocr("立即充值", [0, 0, 10, 10])])
        assert hits == ["充值"]

    def test_no_hits_when_clean(self):
        guard = SafetyGuard(["充值"])
        hits = guard.register_screenshot([ocr("任务", [0, 0, 10, 10])])
        assert hits == []
