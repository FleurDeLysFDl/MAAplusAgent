"""探索记忆：按game_id分命名空间记录已访问过的界面指纹

用于在Observation里提示LLM"这个界面你已经来过N次了"，引导它往未探索的方向走，
避免在同一批界面之间反复横跳。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ExplorationMemory:
    def __init__(self, game_id: str, storage_dir: str = "exploration_logs"):
        self.game_id = game_id
        self.path = Path(storage_dir) / f"{game_id}.json"
        self.visit_counts: dict[str, int] = self._load()

    def _load(self) -> dict[str, int]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.visit_counts, f, ensure_ascii=False, indent=2)

    @staticmethod
    def fingerprint(ocr_texts: list[str]) -> str:
        return hashlib.md5("".join(sorted(ocr_texts)).encode("utf-8")).hexdigest()

    def is_novel(self, ocr_texts: list[str]) -> bool:
        return self.fingerprint(ocr_texts) not in self.visit_counts

    def visit(self, ocr_texts: list[str]) -> tuple[str, int, bool]:
        """记录一次界面访问

        Returns:
            (指纹, 累计访问次数, 是否是本次会话首次见到)
        """
        fp = self.fingerprint(ocr_texts)
        is_novel = fp not in self.visit_counts
        self.visit_counts[fp] = self.visit_counts.get(fp, 0) + 1
        self._save()
        return fp, self.visit_counts[fp], is_novel
