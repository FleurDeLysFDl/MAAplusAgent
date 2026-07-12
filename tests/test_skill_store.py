"""core/skill_store.py的单元测试：技能库的存取和持久化。"""
from __future__ import annotations

from skill_store import LearnedSkill, SkillStore


class TestSkillStore:
    def test_add_and_list(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="去设置", intent_examples=["打开设置"]))
        skills = store.list_all()
        assert len(skills) == 1
        assert skills[0].name == "去设置"
        assert skills[0].success_count == 1

    def test_persists_across_instances(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(
            name="去设置",
            intent_examples=["打开设置"],
            action_sequence=[{"type": "click", "coords": [1, 2], "trigger_text": None}],
        ))

        reloaded = SkillStore("g", storage_dir=str(tmp_path))
        skills = reloaded.list_all()
        assert len(skills) == 1
        assert skills[0].action_sequence == [{"type": "click", "coords": [1, 2], "trigger_text": None}]

    def test_record_success_increments_and_appends_example(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="去设置", intent_examples=["打开设置"]))

        store.record_success("去设置", "帮我去设置界面")

        skill = store.list_all()[0]
        assert skill.success_count == 2
        assert "帮我去设置界面" in skill.intent_examples

    def test_record_success_does_not_duplicate_example(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="去设置", intent_examples=["打开设置"]))
        store.record_success("去设置", "打开设置")

        skill = store.list_all()[0]
        assert skill.intent_examples.count("打开设置") == 1

    def test_record_success_unknown_skill_is_noop(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.record_success("不存在的技能", "任意意图")
        assert store.list_all() == []

    def test_missing_file_starts_empty(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        assert store.list_all() == []
