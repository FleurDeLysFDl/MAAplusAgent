"""MAA-MCP-Server：用mcp SDK (FastMCP) 包装MaaFramework能力，游戏无关。

启动方式（约定从项目根目录运行，profile.yaml里的相对路径都相对于项目根目录）：
    python core/maa_mcp_server.py games/无期迷途/profile.yaml

分工原则：本进程不做任何"下一步该干什么"的决策，只负责"看"（截图/OCR）和"动"
（点击/滑动），以及执行games/<game>/interface.json里声明好的确定性任务。
"""
from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import yaml
from mcp.server.fastmcp import FastMCP

from maa.controller import AdbController
from maa.define import LoggingLevelEnum
from maa.pipeline import JOCR, JRecognitionType
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402
from image_similarity import images_look_same  # noqa: E402
from risk_gate import (  # noqa: E402
    PendingConfirmationQueue,
    RiskGate,
    load_irreversible_keywords,
)
from safety_guard import (  # noqa: E402
    SafetyGuard,
    load_click_forbidden_keywords,
    load_sensitive_keywords,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MaaFramework默认会把Error级别日志直接打到stdout（还带ANSI颜色），
# 但本进程的stdout被MCP用作JSONRPC over stdio的传输通道，两者混在一起会
# 导致客户端解析JSONRPC消息失败。这里关闭stdout日志，改落盘到文件。


def _load_profile(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_adb(profile: dict[str, Any]) -> tuple[str, str]:
    """确定adb可执行文件路径。优先用profile显式声明，否则用Toolkit自动探测"""
    adb_cfg = profile.get("adb", {})
    address = adb_cfg["address"]
    adb_path = adb_cfg.get("adb_path")
    if adb_path:
        return adb_path, address

    devices = Toolkit.find_adb_devices()
    for device in devices:
        if device.address == address:
            return str(device.adb_path), address
    if devices:
        return str(devices[0].adb_path), address

    raise RuntimeError(
        f"找不到adb可执行文件：Toolkit.find_adb_devices()未发现任何设备。"
        f"请在profile.yaml的adb字段里显式声明adb_path，或确认{address}对应的模拟器/真机已开启adb调试"
    )


def _bind_resource(resource: Resource, resource_path: str | list[str]) -> None:
    paths = resource_path if isinstance(resource_path, list) else [resource_path]
    for path in paths:
        job = resource.post_bundle(path).wait()
        if not job.succeeded:
            raise RuntimeError(f"加载资源包失败: {path}")


def build_server(profile_path: str, *, auto_navigate: bool = False) -> FastMCP:
    profile = _load_profile(profile_path)
    game_id = profile["game_id"]

    Tasker.set_stdout_level(LoggingLevelEnum.Off)
    Tasker.set_log_dir(PROJECT_ROOT / "exploration_logs" / "maa_debug" / game_id)

    adb_path, address = _resolve_adb(profile)
    controller = AdbController(adb_path=adb_path, address=address)
    if not controller.post_connection().wait().succeeded:
        raise RuntimeError(f"连接设备失败: {address}（adb_path={adb_path}）")

    resource = Resource()
    _bind_resource(resource, profile["resource_path"])

    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("Tasker绑定Resource/Controller失败")

    with open(profile["interface_path"], "r", encoding="utf-8") as f:
        interface_json = json.load(f)

    safety_guard = SafetyGuard(
        load_sensitive_keywords(profile),
        load_click_forbidden_keywords(profile),
    )
    # auto_navigate=True即自由探索模式（没有用户在场盯着），命中风险的候选点击
    # 跳过+记待确认，不阻塞继续探索（软件探索Agent三大功能模块设计.md模块2.2
    # 场景A）；指令执行模式（auto_navigate=False）本该同步阻塞询问用户（场景B），
    # 还没实现，暂时不启用，risk_gate/pending_queue只在auto_navigate=True时会被
    # 用到，但两个对象本身构造成本很低，两种模式下都创建，不用另外判断
    risk_gate = RiskGate(
        load_sensitive_keywords(profile),
        load_irreversible_keywords(profile),
    )
    pending_queue = PendingConfirmationQueue(game_id)
    memory = ExplorationMemory(game_id)
    image_cache: dict[str, Any] = {}
    # click_on_image只信任"最近一次screenshot()对应的原图"，防止LLM拿一张过期截图上
    # 估算出的坐标去点当前画面（画面可能早就翻篇了）；current_node_id/last_ocr_texts
    # 是给探索记忆的操作记录用的，标记"最近一次screenshot()是哪个节点、看到了什么文字"
    state: dict[str, Any] = {
        "last_image_ref": None,
        "current_node_id": None,
        "last_ocr_texts": [],
    }

    def _trigger_text_at(x: int, y: int) -> str | None:
        """点击坐标落在了哪个OCR候选框里，取它的文字记进操作图（纯尽力而为，找不到就None）"""
        for item in state["last_ocr_texts"]:
            bx, by, bw, bh = item["box"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return item["text"]
        return None

    def _store_image(image: Any) -> str:
        ref = uuid.uuid4().hex
        image_cache[ref] = image
        if len(image_cache) > 20:
            image_cache.pop(next(iter(image_cache)))
        return ref

    def _save_node_image(node_id: str, image: Any) -> None:
        """把这个节点第一次出现时的截图存到nodes/<node_id>.png，跟节点的json文件放一起

        节点json里只有OCR token集合，人工回看探索记录时完全看不出这个节点长什么样；
        只在文件不存在时才写，同一节点后续再访问不重复编码/落盘（is_novel=False的
        重复访问远多于第一次，没必要每次都重新截一份一样的图占IO）。图片编码失败
        不影响整个screenshot()调用——这只是辅助调试用的存档，不是Agent决策要用的
        数据（Agent看图走的是image_ref/get_raw_image那条路）。
        """
        path = memory.nodes_dir / f"{node_id}.png"
        if path.exists():
            return
        ok, buf = cv2.imencode(".png", image)
        if ok:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buf.tobytes())

    # OCR文字相似度落在灰色地带（exploration_memory.ExplorationMemory.
    # SIMILARITY_GRAY_ZONE~SIMILARITY_THRESHOLD之间）时，交给images_look_same
    # 拿像素差异打破僵局（阈值定义见core/image_similarity.py）
    def _images_look_same(node_id: str, image: Any) -> bool:
        path = memory.nodes_dir / f"{node_id}.png"
        if not path.exists():
            return False  # 候选节点没存过图，没法比，保守起见当不同节点处理
        candidate = cv2.imread(str(path))
        if candidate is None:
            return False
        return images_look_same(candidate, image)

    mcp = FastMCP(f"maa-mcp-{game_id}")

    ANDROID_KEYCODE_BACK = 4
    # 自动导航连续跳转的硬上限，防止图数据异常（不该出现的环之类）导致失控；
    # 正常情况下nearest_frontier_path已经保证走的是最短路，远用不到这个上限
    MAX_AUTO_HOPS = 20

    def _box_to_list(box: Any) -> list[int]:
        # MaaFw把all_results/filtered_results/best_result里的box从原始JSON数组
        # 直接塞进dataclass字段，并不会像其它API那样转成Rect对象，所以运行时box
        # 可能是list也可能是Rect，两种都要认
        if isinstance(box, (list, tuple)):
            return [int(v) for v in box]
        return [box.x, box.y, box.w, box.h]

    # 装饰性英文/花体字banner这类OCR认不准的内容，低置信度下会读出完全不成字的
    # 乱码残片（实测同一处banner每次读出的乱码都不一样），混进ocr_texts后既污染
    # 节点匹配用的稳定token集合（把真正稳定的中文标签相似度硬拖低），也污染送给
    # LLM看的screenshot()结果。之前这里用threshold=0.0整个关掉了OCR自带的置信度
    # 过滤（想着"不放过任何文字"），代价是连读不准的噪声也照单全收。改回
    # JOCR默认值0.3，从源头把这类低置信度乱码过滤掉，不必再靠事后调相似度阈值/
    # 图片diff兜底去补救
    OCR_CONFIDENCE_THRESHOLD = 0.3

    def _capture_ocr(image: Any) -> list[dict[str, Any]]:
        ocr_texts: list[dict[str, Any]] = []
        job = tasker.post_recognition(JRecognitionType.OCR, JOCR(threshold=OCR_CONFIDENCE_THRESHOLD), image)
        detail = job.wait().get()
        if detail is not None and detail.nodes:
            reco = detail.nodes[0].recognition
            if reco is not None:
                for r in reco.all_results:
                    text = getattr(r, "text", None)
                    if text is not None:
                        ocr_texts.append({"text": text, "box": _box_to_list(r.box)})
        return ocr_texts

    def _perform_action(action_type: str, coords: list[int] | None) -> None:
        """机械执行一个已知动作的物理操作（不含record_action落盘）。replay_action
        工具和自动导航的每一跳共用这套坐标校验+执行逻辑，两者本质上是同一件事：
        重放一个之前已经验证过坐标的动作，区别只在于是LLM主动选的还是导航自动
        触发的"""
        if action_type == "press_back":
            controller.post_click_key(ANDROID_KEYCODE_BACK).wait()
        elif action_type == "swipe":
            safety_guard.validate_click_on_image(coords[0], coords[1])
            safety_guard.validate_click_on_image(coords[2], coords[3])
            controller.post_swipe(coords[0], coords[1], coords[2], coords[3], 300).wait()
        else:  # click / click_on_image：坐标是历史验证过的，仍然过一遍禁止点击区域
            safety_guard.validate_click_on_image(coords[0], coords[1])
            controller.post_click(coords[0], coords[1]).wait()

    @mcp.tool()
    def screenshot() -> dict:
        """截图，返回结构化信息而非原图：OCR全文+坐标（box为[x,y,w,h]）

        auto_navigate=True（仅自由探索模式下由agent_runner.py开启）时：如果落地的
        画面是一个已知节点、且这个节点已经没有还没试过的候选目标
        （ExplorationMemory.has_frontier为False），会沿着已知有效边自动连续重放
        通向"最近一个还有候选目标"节点的路径，把这段已经摸熟的路直接跳过——返回
        的是跳到头之后那个节点的画面，不是刚落地这个已经摸烂的节点。每一跳仍然
        是一次真实的截图+操作记录落盘（复用_perform_action，跟replay_action走
        同一套物理执行逻辑），只是不需要每一跳都往返一次LLM决策；命中敏感界面、
        或者落地节点跟图里记录的预期不符（游戏状态漂移/意外弹窗）时立刻停止，
        原样返回当时那一步的画面，不勉强继续。
        """
        image = controller.post_screencap().wait().get()
        ocr_texts = _capture_ocr(image)
        sensitive_hits = safety_guard.register_screenshot(ocr_texts)
        node_id, visited_count, is_novel = memory.visit(
            ocr_texts,
            tiebreak=lambda candidate_id: _images_look_same(candidate_id, image),
        )
        _save_node_image(node_id, image)

        auto_navigated_hops = 0
        if auto_navigate and not sensitive_hits and not is_novel and not memory.has_frontier(node_id):
            path = memory.nearest_frontier_path(node_id)
            while path:
                next_id = path[0]
                action = next(
                    (a for a in memory.get_actions(node_id) if a.get("leads_to") == next_id),
                    None,
                )
                if action is None:
                    break  # 图数据和预期对不上，保守放弃剩余的自动导航

                _perform_action(action["type"], action["coords"])
                memory.record_action(
                    node_id, action["type"], action["coords"],
                    trigger_text=action.get("trigger_text"),
                    used_vision=action.get("used_vision", False),
                )

                image = controller.post_screencap().wait().get()
                ocr_texts = _capture_ocr(image)
                sensitive_hits = safety_guard.register_screenshot(ocr_texts)
                node_id, visited_count, is_novel = memory.visit(
                    [t["text"] for t in ocr_texts],
                    tiebreak=lambda candidate_id: _images_look_same(candidate_id, image),
                )
                _save_node_image(node_id, image)
                auto_navigated_hops += 1

                if node_id != next_id or sensitive_hits:
                    break  # 落地节点不是预期的下一跳，或命中敏感界面：立刻停手
                path = path[1:]
                if is_novel or memory.has_frontier(node_id) or auto_navigated_hops >= MAX_AUTO_HOPS:
                    break

        image_ref = _store_image(image)
        state["last_image_ref"] = image_ref
        state["current_node_id"] = node_id
        state["last_ocr_texts"] = ocr_texts
        known_actions = [
            {
                "index": i,
                "type": a["type"],
                "coords": a["coords"],
                "trigger_text": a.get("trigger_text"),
                "attempt_count": a.get("attempt_count", 0),
            }
            for i, a in enumerate(memory.get_actions(node_id))
        ]
        known_ineffective = [
            {"type": a["type"], "coords": a["coords"], "trigger_text": a.get("trigger_text")}
            for a in memory.get_ineffective_actions(node_id)
        ]
        description = memory.nodes.get(node_id, {}).get("description", "")
        result: dict[str, Any] = {
            "ocr_texts": ocr_texts,
            "image_ref": image_ref,
            "is_novel": is_novel,
            "visited_count": visited_count,
            "description": description,
            "known_actions": known_actions,
            "known_ineffective": known_ineffective,
        }
        if auto_navigated_hops:
            result["auto_navigated_hops"] = auto_navigated_hops
        if known_actions:
            result["known_actions_hint"] = (
                "这个界面之前来过，known_actions里是已经验证过坐标的动作，"
                "优先调用replay_action(index)直接执行，不需要重新截图分析或看图定位"
            )
        if known_ineffective:
            result["known_ineffective_hint"] = (
                "known_ineffective里的操作之前试过对这个界面没有效果（点了/按了界面没变），"
                "不要再重复尝试，换别的方式（比如跳过press_back直接用get_raw_image看图找退出方式）"
            )
        # 模板复用：跟当前节点UI布局相似（但内容不同，比如同一个列表模板换了条目、
        # 同一个角色详情页换了角色）的其它节点，把它们身上验证过的点击坐标当参考
        # 提示给出来——不保证在当前节点也有效（click()仍然会按当前画面的OCR候选框
        # 校验坐标），只是省得每换一个内容实例都从头拿视觉摸索一遍
        similar_pairs = memory.similar_nodes(node_id)
        similar_nodes = [
            {
                "node_id": similar_id,
                "similarity": round(score, 2),
                "reference_actions": [
                    {"type": a["type"], "coords": a["coords"], "trigger_text": a.get("trigger_text")}
                    for a in memory.get_actions(similar_id)
                    if a["type"] in ("click", "click_on_image")
                ],
            }
            for similar_id, score in similar_pairs
        ]
        similar_nodes = [s for s in similar_nodes if s["reference_actions"]]
        if similar_nodes:
            result["similar_nodes"] = similar_nodes
            result["similar_nodes_hint"] = (
                "similar_nodes里是UI布局跟当前界面相似（但内容不同）的其它节点，"
                "reference_actions是那些节点上验证过的点击坐标，仅供参考——不保证在"
                "当前界面同样位置也能点到东西，可以在当前ocr_texts候选框里找差不多"
                "位置尝试，但不要跳过校验直接盲点"
            )
        # 提醒Agent考虑mark_node_exhausted()，几种情况都算：
        # 1. untried_tokens已经降到个位数——内容简单的界面，剩下的大概率是装饰文字
        # 2. 已经试过好几个操作、但known_actions（真正有效的）一个都没有——内容
        #    密集的界面（比如角色详情页动辄大几十条OCR文字，untried_tokens几乎
        #    永远降不到个位数）光看"剩多少没试"这条永远不会触发，得看"试了多少次
        #    都白试"，这条信号跟界面文字量无关，能覆盖第1条覆盖不到的密集界面
        # 3. 同一套UI模板换不同内容的一整簇节点（similar_pairs）加起来试过很多次、
        #    没一次有效，或者已经有同伴被标记过exhausted——这类场景每个具体实例
        #    往往只被摸过一两次，单靠自己的历史（第2条）永远攒不够次数，但同一个
        #    模板的其它实例已经反复证明是死路，这本身就是当前节点大概率也是死路
        #    的证据，不该死等这一个节点自己攒够失败次数
        known_ineffective_count = len(memory.get_ineffective_actions(node_id))
        cluster_ids = [node_id] + [sid for sid, _ in similar_pairs]
        cluster_exhausted_siblings = sum(
            1 for sid in cluster_ids[1:] if memory.nodes.get(sid, {}).get("exhausted")
        )
        cluster_effective = sum(len(memory.get_actions(cid)) for cid in cluster_ids)
        cluster_ineffective = sum(len(memory.get_ineffective_actions(cid)) for cid in cluster_ids)
        if (
            not is_novel
            and not memory.nodes.get(node_id, {}).get("exhausted")
            and (
                0 < len(memory.untried_tokens(node_id)) <= 2
                or (known_ineffective_count >= 3 and not known_actions)
                or cluster_exhausted_siblings >= 2
                or (cluster_ineffective >= 5 and cluster_effective == 0)
            )
        ):
            result["exhausted_hint"] = (
                "这个界面已知的候选目标不多了，或者已经试过好几个操作但known_actions"
                "一个有效的都没有，或者跟它同一套UI模板的其它节点已经反复证明是死路"
                "——如果看清楚画面后确认剩下的文字都是标题/公告/资源计数这类点了也"
                "没用的装饰内容、没有还没试过的按钮了，调用mark_node_exhausted()"
                "标记一下，以后自动导航路过这里会直接跳过，不会再把你反复带回来"
            )
        if is_novel:
            result["describe_hint"] = (
                "这是一个新界面，建议调用describe_node(description)简要描述一下这个界面是"
                "干什么的（一两句话即可，比如'兑换中心主界面，可以用材料兑换限定道具'），"
                "方便以后不用重新看图就知道这个节点的用途"
            )
        if sensitive_hits:
            result["sensitive_warning"] = (
                f"检测到敏感界面，命中关键词: {sensitive_hits}。任务终止，不要在此界面执行任何操作。"
            )
        return result

    @mcp.tool()
    def get_raw_image(image_ref: str) -> str:
        """结构化信息不够用时（比如没有文字的图标类按钮、需要点弹窗外空白区域），
        取原图base64给多模态LLM直接看。返回值会被上层agent_runner.py特殊处理成真正
        的图片内容块塞进对话，不是普通文本，不要把这个方法的返回值当文本摘要来读。
        """
        image = image_cache.get(image_ref)
        if image is None:
            raise ValueError(f"image_ref不存在或已过期: {image_ref}")
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @mcp.tool()
    def click_on_image(image_ref: str, x: int, y: int) -> str:
        """在get_raw_image返回的原图上按像素坐标点击。两种场景用这个：
        1. 没有文字、OCR识别不出来的图标类按钮（比如关闭×、返回箭头）
        2. 需要点弹窗/浮层外的空白区域来关闭它（press_back不生效，或者明确要点空白处
           本身而不是"返回上一级"时）

        不校验OCR候选框白名单，只校验image_ref必须是最近一次screenshot()的结果（防止
        对着一张过期截图估算出的坐标去点当前画面）、以及坐标必须落在截图范围内。请只在
        screenshot()的OCR结果里确实找不到目标、并且已经调用get_raw_image看过原图、
        确认了目标位置之后再用这个工具，不要凭空估算坐标。
        """
        if image_ref != state["last_image_ref"]:
            raise ValueError("image_ref不是最近一次screenshot()的结果，请先重新screenshot()")
        image = image_cache.get(image_ref)
        if image is None:
            raise ValueError(f"image_ref不存在或已过期: {image_ref}")
        height, width = image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"坐标({x},{y})超出截图范围({width}x{height})")
        safety_guard.validate_click_on_image(x, y)
        controller.post_click(x, y).wait()
        if state["current_node_id"] is not None:
            memory.record_action(state["current_node_id"], "click_on_image", [x, y], used_vision=True)
        return "clicked"

    @mcp.tool()
    def click(x: int, y: int) -> str:
        """点击坐标，坐标必须落在最近一次screenshot()返回的OCR候选框范围内

        自由探索模式下，如果这个坐标对应的OCR文字命中风险关键词（充值/支付这类
        敏感词，或删除/清空/重置这类不可逆语义），不会真的执行这次点击——跳过，
        记入待确认队列，等用户之后自己审阅决定要不要补做，见core/risk_gate.py。
        """
        safety_guard.validate_click(x, y)
        trigger_text = _trigger_text_at(x, y)

        if auto_navigate and risk_gate.assess(trigger_text) == "high_risk":
            node_id = state["current_node_id"]
            if node_id is not None:
                pending_queue.add(node_id, "click", [x, y], trigger_text, risk_gate.matched_keyword)
            return (
                f"⚠️ 已跳过高危候选「{trigger_text}」（命中「{risk_gate.matched_keyword}」），"
                "不执行这次点击，已记入待确认队列，请换个方向继续探索"
            )

        controller.post_click(x, y).wait()
        if state["current_node_id"] is not None:
            memory.record_action(
                state["current_node_id"], "click", [x, y], trigger_text=trigger_text
            )
        return "clicked"

    @mcp.tool()
    def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
        """滑动，起止坐标都必须落在最近一次screenshot()返回的OCR候选框范围内"""
        safety_guard.validate_swipe(x1, y1, x2, y2)
        controller.post_swipe(x1, y1, x2, y2, duration_ms).wait()
        if state["current_node_id"] is not None:
            memory.record_action(state["current_node_id"], "swipe", [x1, y1, x2, y2])
        return "swiped"

    @mcp.tool()
    def press_back() -> str:
        """发送安卓系统返回键，用于关闭弹窗/浮层/返回上一级界面，不需要猜坐标"""
        controller.post_click_key(ANDROID_KEYCODE_BACK).wait()
        if state["current_node_id"] is not None:
            memory.record_action(state["current_node_id"], "press_back", None)
        return "pressed_back"

    @mcp.tool()
    def describe_node(description: str) -> str:
        """给当前界面（最近一次screenshot()对应的节点）补一句简短描述，比如"兑换中心
        主界面，可以用材料兑换限定道具"。遇到新界面（screenshot()返回is_novel=true）时
        建议调用，方便以后不用重新看图就知道这个节点是干什么的
        """
        node_id = state["current_node_id"]
        if node_id is None:
            raise ValueError("尚未截图，无法记录描述，请先调用screenshot()")
        memory.set_description(node_id, description)
        return "described"

    @mcp.tool()
    def mark_node_exhausted() -> str:
        """确认当前界面（最近一次screenshot()对应的节点）已经没有值得尝试的候选
        目标了——剩下的文字都是标题/公告/资源计数这类点了也没用的装饰内容，没有
        还没试过的按钮/操作。只在你已经看清楚画面、确认过这一点时才调用，不要
        因为懒得再摸而标记。标记之后，自由探索模式下的自动导航路过这个节点时
        会直接跳过，不会再把你反复带回来重新摸索。
        """
        node_id = state["current_node_id"]
        if node_id is None:
            raise ValueError("尚未截图，无法标记，请先调用screenshot()")
        memory.mark_exhausted(node_id)
        return "marked_exhausted"

    @mcp.tool()
    def list_pending_confirmations() -> list[dict]:
        """列出自由探索模式下被跳过的高危候选点击（命中敏感词/不可逆语义，见
        core/risk_gate.py），每条记录了在哪个节点、点了什么文字、命中了哪个
        关键词。这些点击当时没有真的执行，只是记了下来——用户可以照着这份
        清单自己判断要不要手动去补做，Agent本身不会主动replay这些动作。
        """
        return pending_queue.list_all()

    @mcp.tool()
    def replay_action(index: int) -> dict:
        """直接重放当前界面上第index个已知动作（screenshot()返回的known_actions列表里的
        index），坐标是之前探索时已经验证过的，不需要重新截图分析、不需要get_raw_image看图。
        """
        node_id = state["current_node_id"]
        if node_id is None:
            raise ValueError("尚未截图，无法重放动作，请先调用screenshot()")
        actions = memory.get_actions(node_id)
        if not (0 <= index < len(actions)):
            raise ValueError(f"index超出范围，当前界面只有{len(actions)}个已知动作")

        action = actions[index]
        _perform_action(action["type"], action["coords"])

        memory.record_action(
            node_id, action["type"], action["coords"],
            trigger_text=action.get("trigger_text"),
            used_vision=action.get("used_vision", False),
        )
        return {"replayed": action}

    @mcp.tool()
    def list_known_tasks() -> list[dict]:
        """列出当前游戏所有已知确定性任务（复用MBCCtools等现成MaaFramework资源）"""
        return [
            {"name": t["name"], "doc": t.get("doc", "")}
            for t in interface_json.get("task", [])
        ]

    @mcp.tool()
    def run_known_task(task_name: str) -> dict:
        """识别出当前是已知确定性流程（比如每日签到）时直接调用，不用摸索"""
        task_def = next(
            (t for t in interface_json.get("task", []) if t["name"] == task_name),
            None,
        )
        if task_def is None:
            return {"success": False, "task": task_name, "message": f"未知任务: {task_name}"}
        job = tasker.post_task(task_def["entry"]).wait()
        return {"success": job.succeeded, "task": task_name}

    return mcp


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python maa_mcp_server.py <profile.yaml路径> [--auto-navigate]", file=sys.stderr)
        sys.exit(1)

    auto_navigate = "--auto-navigate" in sys.argv[2:]
    server = build_server(sys.argv[1], auto_navigate=auto_navigate)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
