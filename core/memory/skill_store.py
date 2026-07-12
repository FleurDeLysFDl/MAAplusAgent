"""技能库：对应 软件探索Agent三大功能模块设计.md 模块1.3"技能沉淀"的数据层。

这里只有数据结构和存取，不包含"什么时候该把一条成功路径固化成技能"这个决策
逻辑（模块1.3本体的自动沉淀策略，还没实现，见文档开发顺序第5项）——先把存取
接口定下来，core/intent/intent_router.py已经在用它查找已知技能；等1.3落地自动沉淀
时，直接往这个store里add()/record_success()就行，不需要再改数据格式。目前
可以手工调用add()预置一些技能来测试匹配逻辑。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LearnedSkill:
    name: str
    intent_examples: list[str] = field(default_factory=list)
    action_sequence: list[dict[str, Any]] = field(default_factory=list)
    success_count: int = 1


class SkillStore:
    """一个game_id一份json文件（exploration_logs/<game_id>/learned_skills.json），
    跟ExplorationMemory的持久化风格一致（一份跨会话累积的数据，不随进程重启清空）。
    """

    def __init__(self, game_id: str, storage_dir: str = "exploration_logs"):
        self.path = Path(storage_dir) / game_id / "learned_skills.json"
        self.skills: list[LearnedSkill] = self._load()

    def _load(self) -> list[LearnedSkill]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        return [LearnedSkill(**item) for item in raw]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self.skills], f, ensure_ascii=False, indent=2)

    def list_all(self) -> list[LearnedSkill]:
        return list(self.skills)

    def add(self, skill: LearnedSkill) -> None:
        self.skills.append(skill)
        self._save()

    def record_success(self, name: str, intent: str) -> None:
        """已有技能又被相同/相似意图触发一次成功——增加success_count、把这次的
        intent文本也收进intent_examples（扩充以后匹配用的样本池），不新建条目。
        找不到同名技能时什么都不做，调用方应该先用add()新建。
        """
        for skill in self.skills:
            if skill.name == name:
                skill.success_count += 1
                if intent not in skill.intent_examples:
                    skill.intent_examples.append(intent)
                self._save()
                return
