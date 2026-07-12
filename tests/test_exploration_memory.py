"""core/exploration_memory.py的单元测试。

覆盖节点身份匹配（含灰色地带图片diff兜底）、操作图的挂起/落盘、BFS寻路、
模板复用、清理工具这几块核心逻辑。用tmp_path隔离每个测试的存储目录，不碰
真实的exploration_logs。

其中"hub vs tasks不应该被误合并"这个用例是2026-07-12那次mock_game_app实测
踩过的真实回归——SIMILARITY_GRAY_ZONE调低到0之后，两个完全不同的稀疏布局
界面被images_look_same误判成同一屏，这里把当时的真实token数据固化成用例，
以后再调这个阈值不能让这条再挂掉。
"""
from __future__ import annotations

from exploration_memory import (
    ExplorationMemory,
    _boxes_match,
    _is_stable_text,
    _layout_similarity,
    _similarity,
    _stable_items,
    _stable_tokens,
    _tokens_equivalent,
)


def ocr_item(text: str, box: list[int]) -> dict:
    return {"text": text, "box": box}


def ocr_items(pairs: list[tuple[str, list[int]]]) -> list[dict]:
    return [ocr_item(t, b) for t, b in pairs]


# ---------------------------------------------------------------------------
# 纯函数：文字过滤/相似度
# ---------------------------------------------------------------------------


class TestIsStableText:
    def test_pure_digits_rejected(self):
        assert _is_stable_text("123") is False

    def test_pure_symbols_rejected(self):
        assert _is_stable_text("···") is False

    def test_too_short_rejected(self):
        assert _is_stable_text("A") is False

    def test_normal_chinese_text_accepted(self):
        assert _is_stable_text("任务") is True

    def test_mixed_alnum_accepted(self):
        assert _is_stable_text("LV.10") is True


class TestStableTokens:
    def test_filters_noise_and_dedups(self):
        tokens = _stable_tokens(["任务", "1", "任务", "  ", "编队"])
        assert tokens == {"任务", "编队"}


class TestTokensEquivalent:
    def test_exact_match(self):
        assert _tokens_equivalent("任务", "任务") is True

    def test_substring_match(self):
        assert _tokens_equivalent("公告", "查看今日公告") is True
        assert _tokens_equivalent("查看今日公告", "公告") is True

    def test_no_match(self):
        assert _tokens_equivalent("任务", "商城") is False


class TestSimilarity:
    def test_empty_sets_zero(self):
        assert _similarity(set(), {"a"}) == 0.0
        assert _similarity({"a"}, set()) == 0.0

    def test_identical_sets_full_score(self):
        a = {"任务", "编队", "商城"}
        assert _similarity(a, set(a)) == 1.0

    def test_partial_overlap(self):
        # 交集1(任务)，并集3(任务/编队/商城) -> 1/3
        score = _similarity({"任务", "编队"}, {"任务", "商城"})
        assert abs(score - 1 / 3) < 1e-9

    def test_real_false_merge_regression(self):
        """2026-07-12实测：mock_game_app的主城 vs 任务列表，纯文字相似度只有
        0.167（"任务"⊂"每日任务"、"公告"⊂"查看今日公告"两个巧合子串撑起来的），
        这两屏内容完全不同，不应该被判成同一节点。"""
        hub = {"任务", "编队", "角色", "商城", "公告", "测试游戏·主城"}
        tasks = {
            "查看今日公告", "前往", "每日任务", "领取",
            "完成1次日常派遣", "登录游戏", "消耗100点体力", "已完成",
        }
        score = _similarity(hub, tasks)
        assert score < ExplorationMemory.SIMILARITY_GRAY_ZONE


# ---------------------------------------------------------------------------
# 布局相似度（模板复用用）
# ---------------------------------------------------------------------------


class TestLayoutSimilarity:
    def test_same_position_different_text_matches(self):
        """模板复用的核心场景：同一套UI模板换了不同内容，位置几乎不变。"""
        boxes_a = [[10, 20, 40, 20], [10, 50, 40, 20]]
        boxes_b = [[12, 22, 40, 20], [11, 51, 40, 20]]
        assert _layout_similarity(boxes_a, boxes_b) == 1.0

    def test_different_layout_no_match(self):
        boxes_a = [[10, 20, 40, 20]]
        boxes_b = [[900, 700, 40, 20]]
        assert _layout_similarity(boxes_a, boxes_b) == 0.0

    def test_empty_boxes_zero(self):
        assert _layout_similarity([], [[1, 1, 1, 1]]) == 0.0

    def test_boxes_match_size_ratio_bounds(self):
        # 同位置、尺寸差太多（超过2倍）不算同一个控件
        assert _boxes_match([0, 0, 100, 20], [2, 2, 20, 4]) is False
        # 同位置、尺寸接近，算同一个控件
        assert _boxes_match([0, 0, 100, 20], [2, 2, 90, 22]) is True


class TestStableItems:
    def test_keeps_box_alongside_text(self):
        items = _stable_items(ocr_items([("任务", [1, 2, 3, 4]), ("1", [0, 0, 1, 1])]))
        assert items == [{"text": "任务", "box": [1, 2, 3, 4]}]


# ---------------------------------------------------------------------------
# ExplorationMemory.visit()：节点身份匹配
# ---------------------------------------------------------------------------


class TestVisit:
    def test_first_visit_is_novel(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_id, count, is_novel = memory.visit(ocr_items([("任务", [0, 0, 10, 10])]))
        assert is_novel is True
        assert count == 1
        assert node_id in memory.nodes

    def test_revisit_same_content_matches_existing(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        items = ocr_items([("任务", [0, 0, 10, 10]), ("编队", [20, 0, 10, 10])])
        first_id, _, _ = memory.visit(items)
        second_id, count, is_novel = memory.visit(items)
        assert second_id == first_id
        assert is_novel is False
        assert count == 2

    def test_dissimilar_content_creates_new_node(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        first_id, _, _ = memory.visit(ocr_items([("任务", [0, 0, 10, 10])]))
        second_id, _, is_novel = memory.visit(ocr_items([("完全不同的界面文字内容", [0, 0, 10, 10])]))
        assert is_novel is True
        assert second_id != first_id

    def test_gray_zone_tiebreak_true_merges(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        # 精心构造一个落在灰色地带的相似度：3个token对2个，交集1个 -> 1/4=0.25
        first_id, _, _ = memory.visit(ocr_items([("任务", [0, 0, 1, 1]), ("编队", [1, 1, 1, 1])]))
        second_id, _, is_novel = memory.visit(
            ocr_items([("任务", [0, 0, 1, 1]), ("角色", [2, 2, 1, 1]), ("商城", [3, 3, 1, 1])]),
            tiebreak=lambda candidate_id: True,
        )
        assert is_novel is False
        assert second_id == first_id

    def test_gray_zone_tiebreak_false_creates_new(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        first_id, _, _ = memory.visit(ocr_items([("任务", [0, 0, 1, 1]), ("编队", [1, 1, 1, 1])]))
        second_id, _, is_novel = memory.visit(
            ocr_items([("任务", [0, 0, 1, 1]), ("角色", [2, 2, 1, 1]), ("商城", [3, 3, 1, 1])]),
            tiebreak=lambda candidate_id: False,
        )
        assert is_novel is True
        assert second_id != first_id

    def test_new_node_ids_increment_from_disk_max(self, tmp_path):
        """新节点id必须用磁盘上出现过的最大编号+1，不是len(nodes)——见
        ExplorationMemory.__init__的注释，这里验证这个约定真的生效。"""
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        memory.nodes["g-5"] = {
            "id": "g-5", "tokens": ["占位"], "count": 1, "actions": [],
            "description": "", "exhausted": False, "box_signature": [],
        }
        memory._save_node(memory.nodes["g-5"])
        # 重新加载，模拟"曾经有编号5，但0~4从没生成过"的场景
        reloaded = ExplorationMemory("g", storage_dir=str(tmp_path))
        new_id, _, is_novel = reloaded.visit(ocr_items([("全新界面", [0, 0, 1, 1])]))
        assert is_novel is True
        assert new_id == "g-6"


# ---------------------------------------------------------------------------
# 操作记录：record_action + 落盘
# ---------------------------------------------------------------------------


class TestRecordAction:
    def test_pending_action_resolved_on_next_visit(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("界面A", [0, 0, 1, 1])]))
        memory.record_action(node_a, "click", [5, 5], trigger_text="按钮")
        node_b, _, _ = memory.visit(ocr_items([("界面B", [0, 0, 1, 1])]))

        actions = memory.get_actions(node_a)
        assert len(actions) == 1
        assert actions[0]["leads_to"] == node_b
        assert actions[0]["effective"] is True

    def test_self_loop_marked_ineffective(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        items = ocr_items([("界面A", [0, 0, 1, 1])])
        node_a, _, _ = memory.visit(items)
        memory.record_action(node_a, "press_back", None)
        memory.visit(items)  # 还是同一屏 -> 自环

        assert memory.get_actions(node_a) == []  # skip_noop默认过滤掉自环
        ineffective = memory.get_ineffective_actions(node_a)
        assert len(ineffective) == 1
        assert ineffective[0]["leads_to"] == node_a

    def test_overwritten_pending_action_is_dropped(self, tmp_path):
        """挂起后又执行了新操作，旧的挂起记录被直接覆盖丢弃，不强行瞎猜。"""
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("界面A", [0, 0, 1, 1])]))
        memory.record_action(node_a, "click", [1, 1], trigger_text="第一次点击")
        memory.record_action(node_a, "click", [2, 2], trigger_text="第二次点击")
        memory.visit(ocr_items([("界面B", [0, 0, 1, 1])]))

        actions = memory.get_actions(node_a)
        assert len(actions) == 1
        assert actions[0]["trigger_text"] == "第二次点击"

    def test_repeated_action_increments_attempt_count(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        items_a = ocr_items([("界面A", [0, 0, 1, 1])])
        items_b = ocr_items([("界面B", [0, 0, 1, 1])])
        node_a, _, _ = memory.visit(items_a)
        for _ in range(3):
            memory.record_action(node_a, "click", [1, 1], trigger_text="按钮")
            memory.visit(items_b)
            memory.visit(items_a)

        actions = memory.get_actions(node_a)
        assert len(actions) == 1
        assert actions[0]["attempt_count"] == 3


# ---------------------------------------------------------------------------
# untried_tokens / has_frontier / mark_exhausted
# ---------------------------------------------------------------------------


class TestFrontier:
    def test_untried_tokens_excludes_tried_click_targets(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(
            ocr_items([("任务", [0, 0, 1, 1]), ("编队", [1, 1, 1, 1])])
        )
        memory.record_action(node_a, "click", [0, 0], trigger_text="任务")
        memory.visit(ocr_items([("界面B", [0, 0, 1, 1])]))

        untried = memory.untried_tokens(node_a)
        assert untried == {"编队"}

    def test_has_frontier_true_when_untried_remain(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("任务", [0, 0, 1, 1])]))
        assert memory.has_frontier(node_a) is True

    def test_mark_exhausted_overrides_frontier(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("任务", [0, 0, 1, 1])]))
        assert memory.has_frontier(node_a) is True
        memory.mark_exhausted(node_a)
        assert memory.has_frontier(node_a) is False

    def test_mark_exhausted_unknown_node_raises(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        try:
            memory.mark_exhausted("g-999")
            assert False, "应该抛KeyError"
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# BFS寻路：nearest_frontier_path / path_to
# ---------------------------------------------------------------------------


class TestPathfinding:
    def _build_chain(self, tmp_path):
        """搭一条 a -(点按钮)-> b -(点按钮)-> c 的链路，c还有未试候选，a/b没有。"""
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("去B", [0, 0, 1, 1])]))
        memory.record_action(node_a, "click", [0, 0], trigger_text="去B")
        node_b, _, _ = memory.visit(ocr_items([("去C", [0, 0, 1, 1])]))
        memory.record_action(node_b, "click", [0, 0], trigger_text="去C")
        node_c, _, _ = memory.visit(ocr_items([("终点", [0, 0, 1, 1]), ("未试按钮", [1, 1, 1, 1])]))
        memory.mark_exhausted(node_a)
        memory.mark_exhausted(node_b)
        return memory, node_a, node_b, node_c

    def test_nearest_frontier_path_finds_target(self, tmp_path):
        memory, node_a, node_b, node_c = self._build_chain(tmp_path)
        path = memory.nearest_frontier_path(node_a)
        assert path == [node_b, node_c]

    def test_path_to_specific_node(self, tmp_path):
        memory, node_a, node_b, node_c = self._build_chain(tmp_path)
        assert memory.path_to(node_a, node_c) == [node_b, node_c]

    def test_path_to_self_returns_empty_list(self, tmp_path):
        memory, node_a, _, _ = self._build_chain(tmp_path)
        assert memory.path_to(node_a, node_a) == []

    def test_path_to_unreachable_returns_none(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("孤立A", [0, 0, 1, 1])]))
        node_b, _, _ = memory.visit(ocr_items([("孤立B", [0, 0, 1, 1])]))
        assert memory.path_to(node_a, node_b) is None

    def test_nearest_frontier_path_none_when_no_frontier_reachable(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("终点", [0, 0, 1, 1])]))
        memory.mark_exhausted(node_a)
        assert memory.nearest_frontier_path(node_a) is None


# ---------------------------------------------------------------------------
# similar_nodes：模板复用
# ---------------------------------------------------------------------------


class TestSimilarNodes:
    def test_finds_layout_similar_different_content(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("角色A", [10, 20, 40, 20]), ("LV.10", [10, 50, 40, 20])]))
        memory.record_action(node_a, "click", [30, 30], trigger_text="角色A")
        node_b, _, _ = memory.visit(ocr_items([("角色B", [12, 22, 40, 20]), ("LV.20", [11, 51, 40, 20])]))

        similar = memory.similar_nodes(node_b, min_score=0.5)
        assert similar and similar[0][0] == node_a

    def test_no_box_signature_returns_empty(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        memory.nodes["g-0"] = {
            "id": "g-0", "tokens": [], "count": 1, "actions": [],
            "description": "", "exhausted": False, "box_signature": [],
        }
        assert memory.similar_nodes("g-0") == []


# ---------------------------------------------------------------------------
# merge_nodes
# ---------------------------------------------------------------------------


class TestMergeNodes:
    def test_merges_tokens_and_actions(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        keep_id, _, _ = memory.visit(ocr_items([("界面甲", [0, 0, 1, 1])]))
        memory.record_action(keep_id, "click", [0, 0], trigger_text="界面甲")
        other_id, _, _ = memory.visit(ocr_items([("界面乙", [0, 0, 1, 1])]))
        remove_id, _, _ = memory.visit(
            ocr_items([("独有文字C", [0, 0, 1, 1]), ("独有文字D", [1, 1, 1, 1])])
        )
        memory.record_action(remove_id, "click", [5, 5], trigger_text="独有文字D")
        memory.visit(ocr_items([("界面甲", [0, 0, 1, 1])]))  # 回到keep_id，解决挂起

        memory.merge_nodes(keep_id, remove_id)

        assert remove_id not in memory.nodes
        merged = memory.nodes[keep_id]
        assert "独有文字C" in merged["tokens"] and "独有文字D" in merged["tokens"]
        assert any(a.get("trigger_text") == "独有文字D" for a in merged["actions"])

    def test_redirects_dangling_edges_to_keep(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        source_id, _, _ = memory.visit(ocr_items([("来源", [0, 0, 1, 1])]))
        memory.record_action(source_id, "click", [0, 0], trigger_text="按钮")
        remove_id, _, _ = memory.visit(ocr_items([("将被合并", [0, 0, 1, 1])]))
        keep_id, _, _ = memory.visit(ocr_items([("保留", [0, 0, 1, 1])]))

        memory.merge_nodes(keep_id, remove_id)

        source_actions = memory.get_actions(source_id)
        assert source_actions[0]["leads_to"] == keep_id

    def test_same_id_raises(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_id, _, _ = memory.visit(ocr_items([("A", [0, 0, 1, 1])]))
        try:
            memory.merge_nodes(node_id, node_id)
            assert False, "应该抛ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# cleanup / isolated_nodes / delete_node
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_removes_dangling_edges(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("A", [0, 0, 1, 1])]))
        node_a_data = memory.nodes[node_a]
        node_a_data["actions"].append(
            {"type": "click", "coords": [9, 9], "trigger_text": "去往幽灵节点", "leads_to": "g-999", "effective": True, "attempt_count": 1}
        )
        memory._save_node(node_a_data)

        stats = memory.cleanup()
        assert stats["dangling_edges_removed"] == 1
        assert memory.get_actions(node_a) == []

    def test_merges_duplicate_action_entries(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_a, _, _ = memory.visit(ocr_items([("A", [0, 0, 1, 1])]))
        node_b, _, _ = memory.visit(ocr_items([("B", [0, 0, 1, 1])]))
        node_a_data = memory.nodes[node_a]
        node_a_data["actions"] = [
            {"type": "click", "coords": [1, 1], "trigger_text": "按钮", "leads_to": node_b, "effective": True, "attempt_count": 1},
            {"type": "click", "coords": [1, 1], "trigger_text": "按钮", "leads_to": node_b, "effective": True, "attempt_count": 2},
        ]
        memory._save_node(node_a_data)

        stats = memory.cleanup()
        assert stats["duplicate_actions_merged"] == 1
        actions = memory.get_actions(node_a)
        assert len(actions) == 1
        assert actions[0]["attempt_count"] == 3


class TestIsolatedNodes:
    def test_finds_nodes_with_no_edges(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        connected_a, _, _ = memory.visit(ocr_items([("连通A", [0, 0, 1, 1])]))
        memory.record_action(connected_a, "click", [0, 0], trigger_text="按钮")
        memory.visit(ocr_items([("连通B", [0, 0, 1, 1])]))
        isolated, _, _ = memory.visit(ocr_items([("孤立", [0, 0, 1, 1])]))

        result = memory.isolated_nodes()
        assert result == [isolated]

    def test_delete_node_removes_from_memory(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        node_id, _, _ = memory.visit(ocr_items([("待删除", [0, 0, 1, 1])]))
        memory.delete_node(node_id)
        assert node_id not in memory.nodes
        assert not (memory.nodes_dir / f"{node_id}.json").exists()

    def test_delete_unknown_node_raises(self, tmp_path):
        memory = ExplorationMemory("g", storage_dir=str(tmp_path))
        try:
            memory.delete_node("g-999")
            assert False, "应该抛KeyError"
        except KeyError:
            pass
