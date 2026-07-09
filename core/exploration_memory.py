"""探索记忆：按game_id分命名空间记录已访问过的界面签名

用于在Observation里提示LLM"这个界面你已经来过N次了"，引导它往未探索的方向走，
避免在同一批界面之间反复横跳。

不能用"整屏OCR文字拼起来做哈希、要求完全一致才算重复"这种做法——游戏界面里
到处是会跳动的文字（轮播banner、倒计时、资源/货币计数器、OCR对数字的识别
噪声），只要有一个token变了，哈希就完全不同，导致同一个界面几乎每次截图都
被判定成"从没见过"（实测跑了十几轮后，91%的"指纹"只出现过一次）。这里改成
先过滤掉纯数字/纯符号/单字符这类噪声token，再用集合相似度（Jaccard）和
已知界面签名比对，相似度够高就认为是同一个界面。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _stable_tokens(ocr_texts: list[str]) -> set[str]:
    """过滤掉纯数字/纯符号/过短的噪声token，只保留有辨识度的文字标签"""
    tokens: set[str] = set()
    for text in ocr_texts:
        text = text.strip()
        if len(text) < 2:
            continue
        if re.fullmatch(r"[\d\W_]+", text):
            continue
        tokens.add(text)
    return tokens


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ExplorationMemory:
    SIMILARITY_THRESHOLD = 0.6

    def __init__(self, game_id: str, storage_dir: str = "exploration_logs"):
        self.game_id = game_id
        self.path = Path(storage_dir) / f"{game_id}.json"
        self.signatures: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # 旧版{fingerprint: count}格式的数据本身就是脏的（91%只访问过一次，
            # 没有参考价值），直接丢弃重新积累
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.signatures, f, ensure_ascii=False, indent=2)

    def _match(self, tokens: set[str]) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = 0.0
        for sig in self.signatures:
            score = _similarity(tokens, set(sig["tokens"]))
            if score > best_score:
                best_score = score
                best = sig
        return best if best is not None and best_score >= self.SIMILARITY_THRESHOLD else None

    def is_novel(self, ocr_texts: list[str]) -> bool:
        return self._match(_stable_tokens(ocr_texts)) is None

    def visit(self, ocr_texts: list[str]) -> tuple[str, int, bool]:
        """记录一次界面访问

        Returns:
            (界面签名id, 累计访问次数, 是否是本次会话首次见到)
        """
        tokens = _stable_tokens(ocr_texts)
        matched = self._match(tokens)
        if matched is not None:
            matched["count"] += 1
            matched["tokens"] = sorted(tokens)  # 跟着界面缓慢漂移，避免签名过时
            self._save()
            return matched["id"], matched["count"], False

        new_id = f"{self.game_id}-{len(self.signatures)}"
        self.signatures.append({"id": new_id, "tokens": sorted(tokens), "count": 1})
        self._save()
        return new_id, 1, True
