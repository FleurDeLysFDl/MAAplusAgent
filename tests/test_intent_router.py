"""core/intent_router.py的单元测试。

重点覆盖字符2-gram相似度这个替代真LLM语义判断的启发式——包括它在
2026-07-12实机调试时暴露过的"短意图 vs 长描述"不对称问题（用overlap
coefficient而不是Jaccard修复的）。
"""
from __future__ import annotations

from exploration_memory import ExplorationMemory
from intent_router import (
    _char_bigrams,
    _text_similarity,
    match_known_task,
    rank_ocr_candidates_by_relevance,
    route_intent,
    semantic_search_states,
)
from skill_store import LearnedSkill, SkillStore


class TestCharBigrams:
    def test_normal_text(self):
        assert _char_bigrams("任务") == {"任务"}
        assert _char_bigrams("去商城") == {"去商", "商城"}

    def test_single_char_degenerates_to_whole_text(self):
        assert _char_bigrams("去") == {"去"}

    def test_empty_string(self):
        assert _char_bigrams("") == set()


class TestTextSimilarity:
    def test_identical_strings_full_score(self):
        assert _text_similarity("任务界面", "任务界面") == 1.0

    def test_unrelated_strings_low_score(self):
        assert _text_similarity("今天天气怎么样", "商城界面兑换道具") == 0.0

    def test_short_intent_vs_long_description_not_diluted(self):
        """回归用例：2026-07-12实测"去商城兑换东西"对长描述算Jaccard只有0.045
        （被描述里其它无关2-gram稀释），换成overlap coefficient后是0.33。
        用overlap coefficient，短查询命中长文档里的核心词根应该给出较高分数。
        """
        intent = "去商城兑换东西"
        description = "商城界面，显示兑换中心，可以使用契约凭证、经验药剂、强化石、狄斯币等材料兑换各种道具"
        score = _text_similarity(intent, description)
        assert score > 0.3

    def test_empty_string_zero(self):
        assert _text_similarity("", "任何文本") == 0.0


class TestSemanticSearchStates:
    def _memory_with_descriptions(self, tmp_path, descriptions: dict[str, str]) -> ExplorationMemory:
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        for node_id, desc in descriptions.items():
            memory.nodes[node_id] = {
                "id": node_id, "tokens": [], "count": 1, "actions": [],
                "description": desc, "exhausted": False, "box_signature": [],
            }
        return memory

    def test_ranks_best_match_first(self, tmp_path):
        memory = self._memory_with_descriptions(tmp_path, {
            "g-shop": "商城界面，显示兑换中心，可以兑换各种道具",
            "g-tasks": "任务界面，显示每日任务列表",
        })
        results = semantic_search_states("去商城兑换东西", memory)
        assert results[0][0] == "g-shop"

    def test_no_match_returns_empty(self, tmp_path):
        memory = self._memory_with_descriptions(tmp_path, {"g-shop": "商城界面兑换道具"})
        assert semantic_search_states("今天天气怎么样", memory) == []

    def test_nodes_without_description_are_skipped(self, tmp_path):
        memory = self._memory_with_descriptions(tmp_path, {"g-nodesc": ""})
        assert semantic_search_states("任意意图", memory) == []


class TestMatchKnownTask:
    def test_matches_similar_intent_example(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="去设置", intent_examples=["帮我打开设置", "去设置看看"]))
        result = match_known_task("打开设置界面", store)
        assert result is not None
        assert result.name == "去设置"

    def test_no_match_returns_none(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="去设置", intent_examples=["帮我打开设置"]))
        assert match_known_task("今天天气怎么样", store) is None

    def test_empty_store_returns_none(self, tmp_path):
        store = SkillStore("g", storage_dir=str(tmp_path))
        assert match_known_task("任意意图", store) is None


class TestRankOcrCandidatesByRelevance:
    def test_ranks_by_relevance_and_excludes_tried(self):
        ocr_texts = [
            {"text": "任务", "box": [0, 0, 1, 1]},
            {"text": "商城", "box": [1, 1, 1, 1]},
            {"text": "已经试过的按钮", "box": [2, 2, 1, 1]},
        ]
        ranked = rank_ocr_candidates_by_relevance(
            "去商城兑换东西", ocr_texts, tried_texts={"已经试过的按钮"}
        )
        assert [item["text"] for item in ranked] == ["商城", "任务"]


class TestRouteIntent:
    @staticmethod
    def _seed_node_with_description(tmp_path, node_id: str, description: str) -> None:
        """route_intent()内部会重新从磁盘加载ExplorationMemory，直接改内存里的
        memory.nodes不会被它看到，必须真正落盘。"""
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node = {
            "id": node_id, "tokens": [], "count": 1, "actions": [],
            "description": description, "exhausted": False, "box_signature": [],
        }
        memory.nodes[node_id] = node
        memory._save_node(node)

    def test_prefers_known_skill_over_state_search(self, tmp_path):
        self._seed_node_with_description(tmp_path, "g-0", "商城界面兑换道具")
        store = SkillStore("g", storage_dir=str(tmp_path))
        store.add(LearnedSkill(name="goto_shop", intent_examples=["去商城兑换东西"]))

        decision = route_intent("去商城兑换东西", "g", storage_dir=str(tmp_path))
        assert decision["action"] == "run_known_task"
        assert decision["skill"].name == "goto_shop"

    def test_falls_back_to_state_search(self, tmp_path):
        self._seed_node_with_description(tmp_path, "g-0", "商城界面兑换道具")
        decision = route_intent("去商城兑换东西", "g", storage_dir=str(tmp_path))
        assert decision["action"] == "navigate_to_state"
        assert decision["node_id"] == "g-0"

    def test_falls_back_to_probe_when_nothing_matches(self, tmp_path):
        decision = route_intent("今天天气怎么样", "g", storage_dir=str(tmp_path))
        assert decision["action"] == "goal_directed_probe"
