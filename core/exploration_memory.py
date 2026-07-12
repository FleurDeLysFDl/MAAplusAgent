"""探索记忆：按game_id分命名空间，把探索过程记成一张"节点+操作"的图

每个节点是一屏"辨识度够高"的界面（用稳定OCR文字集合的相似度匹配，不要求
完全一致——游戏界面里到处是会跳动的文字，轮播banner/倒计时/资源计数器/OCR
数字识别噪声，只要有一个token变了，精确哈希就完全不同，导致同一个界面几乎
每次截图都被判定成"从没见过"；实测跑了十几轮后91%的"指纹"只出现过一次）。

在这之上叠加"操作"：探索阶段每次click/swipe/click_on_image/press_back成功
执行后，先挂起（这一步的效果要等下一次screenshot()才知道），下次screenshot()
时把"从哪个节点、做了什么操作、到了哪个节点"落盘。图标类操作（原本要靠
get_raw_image+click_on_image才能定位）一旦坐标被记录下来，以后在同一个节点
上重放这个操作就不再需要看图——这是"探索用多模态、复用已知节点用普通模型"
这套双模型路由的数据基础。

存储成一个节点一个json文件（exploration_logs/<game_id>/nodes/<node_id>.json），
方便单独查看/diff每个节点，而不是一个越滚越大的总文件。节点第一次出现时的截图
也存在同一目录下（<node_id>.png，core/maa_mcp_server.py里screenshot()工具负责
落盘），方便人工回看某个节点到底长什么样——光看token集合看不出画面，本类不直接
处理图像数据，image的编码/落盘留给持有cv2依赖的调用方做。

节点还有个description字段（默认空字符串），由Agent遇到新节点时自己调用
set_description()补一句这个界面是干什么的，是为以后"按用途/指令路由到已知
节点"准备的原始数据，路由逻辑本身还没做。
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Callable


def _is_stable_text(text: str) -> bool:
    """过滤掉纯数字/纯符号/过短的噪声文字，只留有辨识度的文字标签"""
    text = text.strip()
    return len(text) >= 2 and not re.fullmatch(r"[\d\W_]+", text)


def _stable_tokens(ocr_texts: list[str]) -> set[str]:
    return {t.strip() for t in ocr_texts if _is_stable_text(t)}


def _stable_items(ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跟_stable_tokens同一套过滤规则，但连box位置一起保留——布局相似度比对
    （见_layout_similarity）要用到位置，只有文字集合不够用"""
    return [
        {"text": item["text"].strip(), "box": item["box"]}
        for item in ocr_items
        if _is_stable_text(item["text"])
    ]


def _box_center(box: list[int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _boxes_match(a: list[int], b: list[int], *, center_tolerance: float = 40) -> bool:
    """两个框算不算"布局上同一个位置"：中心点距离够近，且宽高比例不要差太远

    只看位置和大致尺寸，完全不看框里的文字——这正是模板复用要的效果：同一个
    列表模板换了不同条目内容、同一个角色详情页换了不同角色，位置/尺寸大体不变，
    文字必然不同。容差是像素绝对值，跟具体分辨率有关，profile.yaml里各游戏
    分辨率不一致时可能需要跟着调，暂时先用一个通用值。
    """
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > center_tolerance:
        return False
    aw, ah = max(a[2], 1), max(a[3], 1)
    bw, bh = max(b[2], 1), max(b[3], 1)
    return 0.5 <= aw / bw <= 2.0 and 0.5 <= ah / bh <= 2.0


def _layout_similarity(boxes_a: list[list[int]], boxes_b: list[list[int]]) -> float:
    """改良版Jaccard，跟_similarity同一个贪心两两配对的思路，只是把"文字相不相等"
    换成"位置尺寸像不像"（_boxes_match）。用来近似判断两个节点是不是共享同一套
    UI模板，跟_similarity（判断是不是同一屏）是两件独立的事：模板相似度可以很高，
    同时文字相似度很低（内容完全不同），反之亦然。
    """
    if not boxes_a or not boxes_b:
        return 0.0
    remaining_b = list(boxes_b)
    matched = 0
    for a in boxes_a:
        for i, b in enumerate(remaining_b):
            if _boxes_match(a, b):
                remaining_b.pop(i)
                matched += 1
                break
    union = len(boxes_a) + len(boxes_b) - matched
    return matched / union if union else 0.0


def _tokens_equivalent(x: str, y: str) -> bool:
    """两个token算不算"同一个"：完全相等，或者一个是另一个的子串

    子串这条是为了扛住OCR长文本的box切分不稳定——同一整句公告/描述文字，
    这次识别成一整块，下次可能从中间断成两块，"07月09日停服维"和"07月09日
    停服维护结束公告"内容上是同一句话，精确字符串比较却完全不相等，会把
    Jaccard相似度拉低。要求两边长度都>=2（_stable_tokens已经保证），避免
    单字符之类的短token到处能当子串匹配上，把相似度虚高。
    """
    if x == y:
        return True
    return x in y or y in x


def _similarity(a: set[str], b: set[str]) -> float:
    """改良版Jaccard：交集里的"匹配"允许子串包含，不要求精确相等

    贪心两两配对，不追求最优二分图匹配——这里只是个用来分档的相似度分数，
    不是要精确计数，贪心已经够用。
    """
    if not a or not b:
        return 0.0
    remaining_b = set(b)
    matched = 0
    for x in a:
        for y in remaining_b:
            if _tokens_equivalent(x, y):
                remaining_b.discard(y)
                matched += 1
                break
    union = len(a) + len(b) - matched
    return matched / union if union else 0.0


class ExplorationMemory:
    SIMILARITY_THRESHOLD = 0.6
    # 分数落在[SIMILARITY_GRAY_ZONE, SIMILARITY_THRESHOLD)之间时，文字相似度本身
    # 判断不出"到底是不是同一屏"（实测真实案例：同一个任务列表页只有一个子项从
    # "可用"变"冷却中"，只因为几个图标旁的短噪声token每次OCR不一样，分数刚好卡在
    # 0.583，比0.6差一点点）。这个区间交给visit()的tiebreak回调（通常是调用方拿
    # 图片diff）做最终确认，不传tiebreak就还是老行为：只要没到SIMILARITY_THRESHOLD
    # 一律当新节点
    #
    # 下限先后调过0.45/0.3/0：改到0那次是为了兜住装饰性英文/花体字banner拖低
    # 文字分数的真实案例（wuqimitu-145/146，实测分数0.344），但去掉下限直接
    # 引入了新的误合并——mock_game_app测试app里，主城（图标墙）和任务列表
    # （文字列表）纯文字相似度只有0.167（"任务"⊂"每日任务"、"公告"⊂"查看
    # 今日公告"这两个巧合子串撑起来的），却被images_look_same判成了同一屏：
    # 两屏都是大片空白背景+少量内容，像素diff比例只有2.9%，远低于
    # image_similarity.DIFF_MAX_RATIO(0.35)。这暴露了纯像素diff对"内容稀疏
    # 的界面"天生分辨力不足——背景占比一高，随便两个不同界面都可能长得很像。
    #
    # 下限定在0.25：0.344（该兜的真实重复）在这条线以上，0.167（该拦的误合并）
    # 在这条线以下，两个真实案例正好被这一刀分开。文字相似度极低时说明OCR几乎
    # 认不出任何共同的稳定内容，这时更应该相信"大概率是不同界面"，而不是把
    # 决定权交给对稀疏布局不敏感的像素diff
    SIMILARITY_GRAY_ZONE = 0.25

    def __init__(self, game_id: str, storage_dir: str = "exploration_logs"):
        self.game_id = game_id
        self._base_dir = Path(storage_dir)
        self.nodes_dir = self._base_dir / game_id / "nodes"
        self.nodes: dict[str, dict[str, Any]] = self._load_all()
        # 新节点id必须用"磁盘上出现过的最大编号+1"，不能用len(self.nodes)：
        # 如果历史上有节点丢过（比如曾经的迁移bug、或者未来任何原因导致某个
        # 编号没能落盘），len(nodes)就会比"实际用到过的最大编号"小，下一个
        # 新节点会分配到一个看似空闲、实则已经被别的节点占用的编号，静默覆盖
        # 掉那个已有节点的文件（实测发生过：wuqimitu-0~4丢失后，计数器一直
        # 停留在能撞上wuqimitu-5之后各节点的位置）
        self._next_index = self._compute_next_index()
        self._pending_action: dict[str, Any] | None = None

    def _compute_next_index(self) -> int:
        max_index = -1
        prefix = f"{self.game_id}-"
        for node_id in self.nodes:
            if node_id.startswith(prefix):
                suffix = node_id[len(prefix):]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        return max_index + 1

    def _load_all(self) -> dict[str, dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}
        if self.nodes_dir.exists():
            for path in sorted(self.nodes_dir.glob("*.json")):
                with open(path, "r", encoding="utf-8") as f:
                    node = json.load(f)
                node.setdefault("actions", [])
                node.setdefault("description", "")
                node.setdefault("exhausted", False)
                node.setdefault("box_signature", [])
                nodes[node["id"]] = node

        # 兼容旧版单文件格式（exploration_logs/<game_id>.json，一个list），
        # 迁移成一节点一文件，旧文件不再使用。不能拿"nodes_dir是否已存在"做短路
        # 条件——之前就是这么写的，结果一旦nodes_dir里已经有任意一个节点文件，
        # 这里就整段跳过，legacy文件里那些还没来得及迁移的节点被永久孤立（实测
        # 发生过：wuqimitu-0~4只存在于legacy文件，nodes_dir里已经有5号以后的
        # 节点，导致0~4再也没有机会被读到）。这里改成始终检查legacy文件，只
        # 跳过nodes_dir里已经有同名节点文件的那些id，不重复迁移。
        #
        # 迁移进来的节点必须立刻落盘，不然只存在于这次进程的内存里，这次会话
        # 没有再次访问到就随进程退出彻底丢失（同时下面_compute_next_index()
        # 还是会把它们算进已用编号，之后新建的节点就会分配到实际已经用过、但
        # 没能落盘的编号上，覆盖别的节点——这正是本方法要在这里立刻落盘来避免
        # 的问题）
        legacy_path = self._base_dir / f"{self.game_id}.json"
        if legacy_path.exists():
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for sig in data:
                        if sig["id"] in nodes:
                            continue
                        sig.setdefault("actions", [])
                        sig.setdefault("description", "")
                        sig.setdefault("box_signature", [])
                        nodes[sig["id"]] = sig
                        self._save_node(sig)
            except (json.JSONDecodeError, KeyError):
                pass
        return nodes

    def _save_node(self, node: dict[str, Any]) -> None:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        path = self.nodes_dir / f"{node['id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(node, f, ensure_ascii=False, indent=2)

    def _best_candidate(self, tokens: set[str]) -> tuple[dict[str, Any] | None, float]:
        best: dict[str, Any] | None = None
        best_score = 0.0
        for node in self.nodes.values():
            score = _similarity(tokens, set(node["tokens"]))
            if score > best_score:
                best_score = score
                best = node
        return best, best_score

    def _match(self, tokens: set[str]) -> dict[str, Any] | None:
        best, best_score = self._best_candidate(tokens)
        return best if best is not None and best_score >= self.SIMILARITY_THRESHOLD else None

    def is_novel(self, ocr_texts: list[str]) -> bool:
        return self._match(_stable_tokens(ocr_texts)) is None

    def similarity_between(self, node_id_a: str, node_id_b: str) -> float:
        """两个已知节点之间的文字相似度，给core/dedupe_nodes.py这类离线批量
        比对脚本用——正常的screenshot()->visit()流程走的是_match()内部逻辑，
        不需要这个；这个是事后想重新审视"当时这两个是不是该判成同一个节点"
        时候用的
        """
        a = self.nodes.get(node_id_a)
        b = self.nodes.get(node_id_b)
        if a is None or b is None:
            raise KeyError(f"未知节点: {node_id_a if a is None else node_id_b}")
        return _similarity(set(a["tokens"]), set(b["tokens"]))

    def similar_nodes(
        self, node_id: str, *, min_score: float = 0.5, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """模板复用：找视觉布局（不看文字内容）跟node_id相似的其它节点

        跟_match()/similarity_between()判断"是不是同一屏"是两件独立的事——那边
        比的是文字集合，这里比的是OCR框的位置/尺寸（_layout_similarity），目的
        是揪出"同一套UI模板、换了不同内容"的节点（比如列表的不同条目、角色详情页
        换了不同角色）：这类节点文字相似度可能很低（内容完全不同、不会被判成
        同一个节点），但布局相似度很高，已经在其中一个节点上验证过的按钮坐标，
        大概率在同模板的其它节点上也能点到差不多的位置，可以当参考提示给Agent，
        省得每换一个内容实例都从头拿视觉摸索一遍坐标。

        返回按相似度降序的[(node_id, score), ...]，不含自己，最多top_k个。
        """
        target = self.nodes.get(node_id)
        if target is None or not target.get("box_signature"):
            return []
        scored = []
        for other_id, other in self.nodes.items():
            if other_id == node_id or not other.get("box_signature"):
                continue
            score = _layout_similarity(target["box_signature"], other["box_signature"])
            if score >= min_score:
                scored.append((other_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def visit(
        self,
        ocr_items: list[dict[str, Any]],
        *,
        tiebreak: Callable[[str], bool] | None = None,
    ) -> tuple[str, int, bool]:
        """记录一次界面访问，顺带把上一次record_action()挂起的操作落盘

        ocr_items: 当前截图的OCR结果，每项{"text": str, "box": [x,y,w,h]}——
        不只是文字，box位置也要存下来（存进节点的box_signature字段），给
        similar_nodes()的布局相似度比对用。

        tiebreak: 可选回调，只在文字相似度落在[SIMILARITY_GRAY_ZONE,
        SIMILARITY_THRESHOLD)这个"像但判断不了"的区间时才会被调用，参数是候选
        节点id，返回True表示确认就是这个节点、False表示当成新节点处理。不传
        （默认None）就是老行为：只用SIMILARITY_THRESHOLD一刀切。本类不直接
        处理图像数据，调用方（比如core/maa_mcp_server.py，手头有当前截图和
        candidate节点存的png）可以传一个做图片diff的函数进来，用完全独立于
        文字匹配的信号打破僵局。

        Returns:
            (节点id, 累计访问次数, 是否是本次会话首次见到)
        """
        stable_items = _stable_items(ocr_items)
        tokens = {item["text"] for item in stable_items}
        box_signature = [item["box"] for item in stable_items]
        matched = self._match(tokens)
        if matched is None and tiebreak is not None:
            candidate, score = self._best_candidate(tokens)
            if candidate is not None and self.SIMILARITY_GRAY_ZONE <= score < self.SIMILARITY_THRESHOLD:
                if tiebreak(candidate["id"]):
                    matched = candidate
        if matched is not None:
            matched["count"] += 1
            matched["tokens"] = sorted(tokens)  # 跟着界面缓慢漂移，避免签名过时
            matched["box_signature"] = box_signature
            node_id, count, is_novel = matched["id"], matched["count"], False
        else:
            node_id = f"{self.game_id}-{self._next_index}"
            self._next_index += 1
            matched = {
                "id": node_id,
                "tokens": sorted(tokens),
                "count": 1,
                "actions": [],
                "description": "",
                "exhausted": False,
                "box_signature": box_signature,
            }
            self.nodes[node_id] = matched
            count, is_novel = 1, True

        if self._pending_action is not None:
            self._resolve_pending(node_id)
        self._save_node(matched)
        return node_id, count, is_novel

    def record_action(
        self,
        from_node_id: str,
        action_type: str,
        coords: list[int] | None,
        trigger_text: str | None = None,
        used_vision: bool = False,
    ) -> None:
        """click/swipe/click_on_image/press_back成功执行后调用

        这一步的效果要等下一次screenshot()才知道（导向了哪个节点），所以先挂起，
        由下一次visit()负责落盘。如果挂起后没等到下一次screenshot()又执行了新的
        操作，旧的挂起记录会被直接覆盖丢弃（不强行瞎猜落到了哪个节点）。
        """
        self._pending_action = {
            "from_node": from_node_id,
            "action": {
                "type": action_type,
                "coords": coords,
                "trigger_text": trigger_text,
                "used_vision": used_vision,
            },
        }

    def _resolve_pending(self, to_node_id: str) -> None:
        pending = self._pending_action
        self._pending_action = None
        from_node = self.nodes.get(pending["from_node"])
        if from_node is None:
            return

        action = pending["action"]
        # leads_to指向自己=点了但界面没变，说明这个操作在这个节点上没用（比如
        # press_back对这一整类全屏浮层不生效）。这个信息本身也很值钱：不把它
        # 存下来的话，同一个已知没用的操作会一次又一次被重试，白白浪费步数
        effective = to_node_id != from_node["id"]
        for existing in from_node["actions"]:
            if existing["type"] == action["type"] and existing["coords"] == action["coords"]:
                existing["leads_to"] = to_node_id
                existing["effective"] = effective
                existing["attempt_count"] = existing.get("attempt_count", 0) + 1
                self._save_node(from_node)
                return

        action["leads_to"] = to_node_id
        action["effective"] = effective
        action["attempt_count"] = 1
        from_node["actions"].append(action)
        self._save_node(from_node)

    def get_actions(self, node_id: str, *, skip_noop: bool = True) -> list[dict[str, Any]]:
        """给复用阶段用：这个节点上已知能执行、且确认过有效果（不是原地不动）的操作

        skip_noop=True时过滤掉effective=False的操作（点了但界面没变，对导航没
        意义，但原始记录里仍然保留，不影响探索阶段继续观察它，也保留给
        get_ineffective_actions()用）
        """
        node = self.nodes.get(node_id)
        if node is None:
            return []
        actions = [a for a in node["actions"] if "leads_to" in a]
        if skip_noop:
            actions = [a for a in actions if a.get("effective", a["leads_to"] != node_id)]
        return actions

    def get_ineffective_actions(self, node_id: str) -> list[dict[str, Any]]:
        """给"别重复重试"用：这个节点上已知点了/滑了/按了没反应的操作

        不管探索还是复用模式都该看到这个（跟known_actions那条"抄近道"的捷径
        不一样，这条是纯负面信息，不会导致在已知节点之间打转，只会少走弯路）
        """
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [
            a for a in node["actions"]
            if "leads_to" in a and not a.get("effective", a["leads_to"] != node_id)
        ]

    def merge_nodes(self, keep_id: str, remove_id: str) -> None:
        """把remove_id合并进keep_id，用于人工修正"明明是同一屏、却被OCR相似度判定成
        两个节点"的情况（比如同一个列表页只有一个子项从"可用"变成"冷却中"，短噪声token
        把Jaccard相似度拉到刚好卡在阈值以下）。self._match()本身不做这个判断，这个方法
        只处理"已经知道要合并"之后的数据搬运：

        1. token集合取并集（以后匹配时两边的样子都认）
        2. count相加
        3. description优先保留keep_id的，keep_id没写过才用remove_id的
        4. actions合并，按(type, coords)去重（跟_resolve_pending同一套判重逻辑），
           重复的把attempt_count相加、effective取或
        5. 全图扫一遍其它节点，把leads_to==remove_id的边改指向keep_id——这一步不做
           的话，remove_id删掉后这些边会变成指向不存在节点的悬空引用
        6. remove_id的截图如果keep_id还没有，搬过去；json文件直接删掉
        """
        keep = self.nodes.get(keep_id)
        remove = self.nodes.get(remove_id)
        if keep is None or remove is None:
            raise KeyError(f"未知节点: keep={keep_id} remove={remove_id}")
        if keep_id == remove_id:
            raise ValueError("keep_id和remove_id不能是同一个节点")

        keep["tokens"] = sorted(set(keep["tokens"]) | set(remove["tokens"]))
        keep["count"] = keep.get("count", 0) + remove.get("count", 0)
        if not keep.get("description") and remove.get("description"):
            keep["description"] = remove["description"]

        for action in remove.get("actions", []):
            if action.get("leads_to") == remove_id:
                action["leads_to"] = keep_id  # 指向自己的无效边，合并后指向新的自己
            merged = False
            for existing in keep["actions"]:
                if existing["type"] == action["type"] and existing["coords"] == action["coords"]:
                    existing["attempt_count"] = existing.get("attempt_count", 0) + action.get(
                        "attempt_count", 1
                    )
                    existing["effective"] = existing.get("effective", False) or action.get(
                        "effective", False
                    )
                    merged = True
                    break
            if not merged:
                keep["actions"].append(action)

        for node_id, node in self.nodes.items():
            if node_id in (keep_id, remove_id):
                continue
            redirected = False
            for action in node.get("actions", []):
                if action.get("leads_to") == remove_id:
                    action["leads_to"] = keep_id
                    redirected = True
            if redirected:
                self._save_node(node)

        remove_png = self.nodes_dir / f"{remove_id}.png"
        keep_png = self.nodes_dir / f"{keep_id}.png"
        if remove_png.exists():
            if keep_png.exists():
                remove_png.unlink()
            else:
                remove_png.rename(keep_png)

        (self.nodes_dir / f"{remove_id}.json").unlink(missing_ok=True)
        del self.nodes[remove_id]
        self._save_node(keep)

    def cleanup(self) -> dict[str, int]:
        """整理已落盘的节点/操作数据，处理探索过程中可能积累的两类一致性问题：

        1. 悬空边：action.leads_to指向一个已经不存在的节点id（比如node.json被
           手动删除、或者merge_nodes()覆盖不到的历史遗留问题）。这类action已经
           没有意义的目标，直接删除——留着的话，nearest_frontier_path()之类的
           图算法会读到一条查无此节点的边，白白浪费比对
        2. 重复动作：同一个节点上出现(type, coords)完全相同的action条目不止
           一条（正常走record_action()->_resolve_pending()会自动合并，理论上
           不该出现，但历史数据/手工编辑可能留下这种重复）。保留第一条，
           attempt_count相加、effective取或，其余删除

        只处理数据内部一致性问题，不删除/合并节点本身——孤立节点可能是还没找到
        路径进去的真实界面，删不删需要人工判断（见visualize_graph.py的isolated
        标记），跟merge_nodes()/dedupe_nodes.py处理的"这两个节点该不该算同一个"
        是不同的问题，这个方法不碰。

        Returns:
            {"dangling_edges_removed": 删除的悬空边数, "duplicate_actions_merged": 合并掉的重复动作数}
        """
        dangling_removed = 0
        duplicates_merged = 0
        for node in self.nodes.values():
            original_count = len(node.get("actions", []))
            kept: list[dict[str, Any]] = []
            seen: dict[tuple[str, str], dict[str, Any]] = {}
            for action in node.get("actions", []):
                leads_to = action.get("leads_to")
                if leads_to is not None and leads_to not in self.nodes:
                    dangling_removed += 1
                    continue
                key = (action["type"], json.dumps(action.get("coords")))
                existing = seen.get(key)
                if existing is not None:
                    existing["attempt_count"] = existing.get("attempt_count", 1) + action.get(
                        "attempt_count", 1
                    )
                    existing["effective"] = existing.get("effective", False) or action.get(
                        "effective", False
                    )
                    duplicates_merged += 1
                    continue
                seen[key] = action
                kept.append(action)
            if len(kept) != original_count:
                node["actions"] = kept
                self._save_node(node)
        return {
            "dangling_edges_removed": dangling_removed,
            "duplicate_actions_merged": duplicates_merged,
        }

    def isolated_nodes(self) -> list[str]:
        """找出既无入边也无出边的节点——可能是OCR噪声偶然被判成新节点的垃圾数据，
        也可能是还没找到路径进去的真实存在的界面（比如探索时用get_raw_image/
        click_on_image直接跳过去、没有留下可重放的普通click记录），删不删需要
        人工判断，这个方法只负责找出来，不负责删（跟visualize_graph.py的isolated
        标记是同一个概念，删除动作见delete_node）。
        """
        incoming = {nid: 0 for nid in self.nodes}
        for node_id in self.nodes:
            for action in self.get_actions(node_id):
                target = action.get("leads_to")
                if target in incoming:
                    incoming[target] += 1
        return [
            nid for nid in self.nodes
            if incoming.get(nid, 0) == 0 and len(self.get_actions(nid)) == 0
        ]

    def delete_node(self, node_id: str) -> None:
        """删除一个节点及其截图文件。调用方（比如cleanup_graph.py --delete-isolated）
        负责判断这个节点是不是真的该删，这个方法本身不做判断、只负责执行。不检查
        是否有其它节点的action.leads_to指向它——那样会产生悬空边，留给cleanup()
        下次运行时清理，不在这里现场处理（避免这个方法職责过重）。
        """
        if node_id not in self.nodes:
            raise KeyError(f"未知节点: {node_id}")
        (self.nodes_dir / f"{node_id}.png").unlink(missing_ok=True)
        (self.nodes_dir / f"{node_id}.json").unlink(missing_ok=True)
        del self.nodes[node_id]

    def untried_tokens(self, node_id: str) -> set[str]:
        """这个节点上，OCR文字里还没被任何点击类操作覆盖过的部分——粗略代表"看得见
        但还没试过的候选目标"。只统计click/click_on_image留下的trigger_text：
        swipe/press_back没有关联的文字，没法用这种方式判断"是否还有新意"，这类
        操作值不值得再试，仍然只能靠Agent自己判断，不在这个方法的能力范围内。

        这是个近似：trigger_text之外的稳定token也可能只是装饰性文字/标题，不一定
        真的可点。误判的代价很小——调用方（比如BFS导航）把Agent带到这种"假前沿"
        节点后，Agent自己一看画面就知道没什么可点的，正常继续摸索，不会卡死。
        """
        node = self.nodes.get(node_id)
        if node is None:
            return set()
        tried = {a["trigger_text"] for a in node["actions"] if a.get("trigger_text")}
        return {
            token for token in node["tokens"]
            if not any(_tokens_equivalent(token, t) for t in tried)
        }

    def has_frontier(self, node_id: str) -> bool:
        """这个节点上是否还有没试过的候选目标

        优先看exhausted标记（见mark_exhausted）：很多OCR文字是标题/公告/资源计数
        这类永远不会被点的装饰内容，untried_tokens只能近似猜、猜不准就会导致这个
        节点永远"看起来还有前沿"、BFS自动导航永远绕不开它。Agent自己看画面确认过
        "真的没什么可点了"之后调用mark_node_exhausted()，比继续猜token准，这里
        直接采信、不再看untried_tokens。
        """
        node = self.nodes.get(node_id)
        if node is not None and node.get("exhausted"):
            return False
        return bool(self.untried_tokens(node_id))

    def mark_exhausted(self, node_id: str) -> None:
        """人工确认（通常是Agent自己看画面判断）这个节点已经没有值得尝试的候选
        目标了，覆盖untried_tokens的启发式猜测。标记之后has_frontier()对这个
        节点恒为False，BFS自动导航会正常把它当"已摸透"跳过。
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"未知节点: {node_id}")
        node["exhausted"] = True
        self._save_node(node)

    def _bfs_path(self, from_node_id: str, goal: Callable[[str], bool]) -> list[str] | None:
        """在已知的"有效边"(action.leads_to且effective)构成的图上做BFS，找到从
        from_node_id出发最近的一个满足goal(node_id)的节点，返回路径（不含起点、
        含终点，按经过顺序排列）。找不到（比如压根没有已知边可走、或可达范围内
        没有满足goal的节点）返回None。nearest_frontier_path/path_to共用这同一套
        搜索，区别只是goal条件不同。
        """
        if from_node_id not in self.nodes:
            return None
        visited = {from_node_id}
        queue: deque[tuple[str, list[str]]] = deque([(from_node_id, [])])
        while queue:
            node_id, path = queue.popleft()
            for action in self.get_actions(node_id):
                target = action.get("leads_to")
                if target is None or target in visited or target not in self.nodes:
                    continue
                new_path = path + [target]
                if goal(target):
                    return new_path
                visited.add(target)
                queue.append((target, new_path))
        return None

    def nearest_frontier_path(self, from_node_id: str) -> list[str] | None:
        """从from_node_id出发，最近的一个有未试候选(has_frontier)的节点的路径。

        起点自己是否有frontier不在这里判断——调用方应该先查
        has_frontier(from_node_id)，起点本身就有的话不需要导航。这个方法只负责
        "在已知连通性里找最短路"，不负责"值不值得走这条路"之类的判断，那些
        决策留给调用方。
        """
        return self._bfs_path(from_node_id, self.has_frontier)

    def path_to(self, from_node_id: str, to_node_id: str) -> list[str] | None:
        """从from_node_id出发到指定to_node_id的最短已知路径（不含起点、含终点）。
        找不到（比如根本没有已知边可达）返回None；起点等于终点返回空列表。

        给core/navigate_demo.py这类"导航到某个具体已知节点"的场景用——跟
        nearest_frontier_path目标不一样（那个是"随便找个还没摸完的"，这个是
        "必须是这一个"），共用同一套_bfs_path搜索。
        """
        if from_node_id == to_node_id:
            return []
        return self._bfs_path(from_node_id, lambda node_id: node_id == to_node_id)

    def set_description(self, node_id: str, description: str) -> None:
        """给节点补一句人话描述，比如"兑换中心主界面，可以用材料兑换限定道具"

        目前只管把数据存下来——由Agent自己在遇到新节点(is_novel)时主动调用，给以后
        "不用重新看OCR token/截图就知道这个节点是干什么的"打个基础，为后续按用途/
        指令路由到已知节点做准备；路由本身怎么做（比如按描述文本做相似度匹配）还没
        实现，不在这个方法的职责范围内。
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"未知节点: {node_id}")
        node["description"] = description.strip()
        self._save_node(node)
