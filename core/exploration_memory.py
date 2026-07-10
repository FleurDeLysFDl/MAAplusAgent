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

    def _match(self, tokens: set[str]) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = 0.0
        for node in self.nodes.values():
            score = _similarity(tokens, set(node["tokens"]))
            if score > best_score:
                best_score = score
                best = node
        return best if best is not None and best_score >= self.SIMILARITY_THRESHOLD else None

    def is_novel(self, ocr_texts: list[str]) -> bool:
        return self._match(_stable_tokens(ocr_texts)) is None

    def visit(self, ocr_texts: list[str]) -> tuple[str, int, bool]:
        """记录一次界面访问，顺带把上一次record_action()挂起的操作落盘

        Returns:
            (节点id, 累计访问次数, 是否是本次会话首次见到)
        """
        tokens = _stable_tokens(ocr_texts)
        matched = self._match(tokens)
        if matched is not None:
            matched["count"] += 1
            matched["tokens"] = sorted(tokens)  # 跟着界面缓慢漂移，避免签名过时
            node_id, count, is_novel = matched["id"], matched["count"], False
        else:
            node_id = f"{self.game_id}-{self._next_index}"
            self._next_index += 1
            matched = {"id": node_id, "tokens": sorted(tokens), "count": 1, "actions": []}
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
