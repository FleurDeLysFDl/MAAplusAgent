"""高危操作确认Harness：对应 软件探索Agent三大功能模块设计.md 模块2。

跟core/safety/safety_guard.py的分工不是一回事：
- SafetyGuard做的是硬校验/整屏拦截——坐标越界、命中click_forbidden区域直接拒绝
  执行，整屏OCR命中敏感词直接告警终止会话。这三种情况都不该被绕过，跟"用户是否
  在场"无关。
- RiskGate做的是"这一个候选点击值不值得直接执行"的风险评估，只看候选自己的
  目标文字，不整屏拦截；命中风险时的处理策略取决于当前是不是有用户在场：
  自由探索模式（没人盯着）下跳过、记入待确认队列，不阻塞继续探索（本模块
  当前只实现这一种场景，对应文档2.2场景A）；指令执行模式下本该同步阻塞询问
  用户（文档2.2场景B），这个场景还没实现，指令模式暂时不启用RiskGate，沿用
  原有行为。

跳过的候选记进本地持久化的待确认队列（exploration_logs/<game_id>/
pending_confirmations.json），等用户下次上线自己审阅决定要不要补做。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 不可逆语义关键词：跟safety_guard.GLOBAL_SENSITIVE_KEYWORDS（充值/支付这类"钱"
# 相关）是两类不同的风险，这里收录的是"点了很难/不能撤销"的操作语义。故意不收
# "确定""确认""提交"这类词——它们太通用，日常关闭弹窗/提交搜索都会用到，收进来
# 只会疯狂误报，范围比敏感词更窄、更保守
DEFAULT_IRREVERSIBLE_KEYWORDS = ["删除", "清空", "重置", "解绑", "注销", "退出登录", "格式化"]


def load_irreversible_keywords(profile: dict[str, Any]) -> list[str]:
    """合并内置的不可逆语义词与profile里声明的游戏专属补充（同一份
    sensitive_keywords.yaml文件里的irreversible字段，跟extra/click_forbidden
    并列，不需要新开一个配置文件）"""
    extra: list[str] = []
    path = profile.get("sensitive_keywords_path")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            extra = data.get("irreversible", [])
        except FileNotFoundError:
            pass
    return DEFAULT_IRREVERSIBLE_KEYWORDS + extra


class RiskGate:
    """评估单个候选点击目标的风险等级，只看目标文字本身，不看整屏。"""

    def __init__(self, sensitive_keywords: list[str], irreversible_keywords: list[str]):
        self.sensitive_keywords = sensitive_keywords
        self.irreversible_keywords = irreversible_keywords
        self.matched_keyword: str | None = None

    def assess(self, target_text: str | None) -> str:
        """返回"high_risk"或"safe"。target_text为None（比如click_on_image这类
        没有OCR文字的图标点击）时没有可判断的文字依据，一律当safe——图标类
        点击的风险评估不在这个方法的能力范围内，也超出了本模块当前的实现范围。
        """
        self.matched_keyword = None
        if not target_text:
            return "safe"
        for kw in self.sensitive_keywords:
            if kw in target_text:
                self.matched_keyword = kw
                return "high_risk"
        for kw in self.irreversible_keywords:
            if kw in target_text:
                self.matched_keyword = kw
                return "high_risk"
        return "safe"


class PendingConfirmationQueue:
    """自由探索模式下被RiskGate跳过的候选动作，持久化存一份，等用户之后审阅。

    存成一个game_id一份json文件，跟ExplorationMemory一节点一文件的做法不是
    一回事——待确认队列量通常不大，不需要拆文件。
    """

    def __init__(self, game_id: str, storage_dir: str = "exploration_logs"):
        self.path = Path(storage_dir) / game_id / "pending_confirmations.json"
        self.items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def add(
        self,
        node_id: str,
        action_type: str,
        coords: list[int] | None,
        trigger_text: str | None,
        matched_keyword: str,
    ) -> None:
        self.items.append(
            {
                "node_id": node_id,
                "type": action_type,
                "coords": coords,
                "trigger_text": trigger_text,
                "matched_keyword": matched_keyword,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._save()

    def list_all(self) -> list[dict[str, Any]]:
        return list(self.items)

    def remove(self, index: int) -> dict[str, Any]:
        """移除并返回index位置的待确认项——放行执行后，或者用户审阅完决定不管了，
        都要从队列里摘掉，不然下次审阅还会重复看到同一条（见core/tools/review_pending.py）"""
        item = self.items.pop(index)
        self._save()
        return item

    def clear(self) -> int:
        """一次性清空整个队列（用户审阅后决定全部丢弃），返回清空前的条目数"""
        n = len(self.items)
        self.items = []
        self._save()
        return n
