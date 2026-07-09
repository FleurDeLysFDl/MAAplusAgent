"""安全护栏：坐标白名单校验 + 敏感词拦截

护栏做在MCP Server层，不依赖Prompt约束LLM的行为。
"""
from __future__ import annotations

from typing import Any

import yaml

GLOBAL_SENSITIVE_KEYWORDS = ["支付", "实名认证", "绑定手机", "删除账号", "充值"]


def load_sensitive_keywords(profile: dict[str, Any]) -> list[str]:
    """合并全局敏感词与profile里声明的游戏专属敏感词（整屏拦截，命中即终止探索）"""
    extra: list[str] = []
    path = profile.get("sensitive_keywords_path")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            extra = data.get("extra", [])
        except FileNotFoundError:
            pass
    return GLOBAL_SENSITIVE_KEYWORDS + extra


def load_click_forbidden_keywords(profile: dict[str, Any]) -> list[str]:
    """游戏专属的"禁止点击"关键词：不整屏拦截，只屏蔽命中这些关键词的OCR候选框附近"""
    path = profile.get("sensitive_keywords_path")
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return data.get("click_forbidden", [])


class CoordinateRejected(Exception):
    """坐标不在最近一次screenshot()返回的OCR候选框范围内，或落在禁止点击的区域附近"""


class SafetyGuard:
    """MCP Server层安全护栏

    - 坐标白名单：click/swipe坐标必须落在最近一次screenshot()的OCR候选框内
    - 敏感界面拦截：screenshot()返回的OCR全文命中敏感词时直接告警，不交给LLM判断
    - 禁止点击区域：命中click_forbidden关键词的OCR候选框及其附近一圈，任何点击/滑动
      方式（包括绕过白名单的click_on_image）都不允许落在这个区域内
    """

    FORBIDDEN_PADDING_PX = 60

    def __init__(self, sensitive_keywords: list[str], click_forbidden_keywords: list[str] | None = None):
        self.sensitive_keywords = sensitive_keywords
        self.click_forbidden_keywords = click_forbidden_keywords or []
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._forbidden_boxes: list[tuple[int, int, int, int]] = []

    def register_screenshot(self, ocr_texts: list[dict[str, Any]]) -> list[str]:
        """每次screenshot()调用后登记最新的OCR候选框，返回命中的敏感词（可能为空）"""
        self._last_boxes = [tuple(r["box"]) for r in ocr_texts]
        self._forbidden_boxes = [
            tuple(r["box"])
            for r in ocr_texts
            if any(kw in r["text"] for kw in self.click_forbidden_keywords)
        ]
        joined_text = "".join(r["text"] for r in ocr_texts)
        return [kw for kw in self.sensitive_keywords if kw in joined_text]

    def _point_in_any_box(self, x: int, y: int) -> bool:
        return any(
            bx <= x <= bx + bw and by <= y <= by + bh
            for bx, by, bw, bh in self._last_boxes
        )

    def _point_near_forbidden_box(self, x: int, y: int) -> bool:
        pad = self.FORBIDDEN_PADDING_PX
        return any(
            bx - pad <= x <= bx + bw + pad and by - pad <= y <= by + bh + pad
            for bx, by, bw, bh in self._forbidden_boxes
        )

    def _check_forbidden(self, x: int, y: int) -> None:
        if self._point_near_forbidden_box(x, y):
            raise CoordinateRejected(
                f"坐标({x},{y})落在禁止点击的区域（{self.click_forbidden_keywords}）附近，拒绝执行"
            )

    def validate_click(self, x: int, y: int) -> None:
        if not self._last_boxes:
            raise CoordinateRejected("尚未截图，无法校验点击坐标，请先调用screenshot()")
        if not self._point_in_any_box(x, y):
            raise CoordinateRejected(
                f"点击坐标({x},{y})不在最近一次screenshot()的OCR候选框范围内，拒绝执行"
            )
        self._check_forbidden(x, y)

    def validate_swipe(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._last_boxes:
            raise CoordinateRejected("尚未截图，无法校验滑动坐标，请先调用screenshot()")
        if not self._point_in_any_box(x1, y1):
            raise CoordinateRejected(
                f"滑动起点({x1},{y1})不在最近一次screenshot()的OCR候选框范围内，拒绝执行"
            )
        if not self._point_in_any_box(x2, y2):
            raise CoordinateRejected(
                f"滑动终点({x2},{y2})不在最近一次screenshot()的OCR候选框范围内，拒绝执行"
            )
        self._check_forbidden(x1, y1)
        self._check_forbidden(x2, y2)

    def validate_click_on_image(self, x: int, y: int) -> None:
        """click_on_image专用：不校验OCR候选框白名单，但仍要遵守禁止点击区域"""
        self._check_forbidden(x, y)
