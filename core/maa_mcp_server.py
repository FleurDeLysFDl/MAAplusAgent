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


def build_server(profile_path: str) -> FastMCP:
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

    mcp = FastMCP(f"maa-mcp-{game_id}")

    def _box_to_list(box: Any) -> list[int]:
        # MaaFw把all_results/filtered_results/best_result里的box从原始JSON数组
        # 直接塞进dataclass字段，并不会像其它API那样转成Rect对象，所以运行时box
        # 可能是list也可能是Rect，两种都要认
        if isinstance(box, (list, tuple)):
            return [int(v) for v in box]
        return [box.x, box.y, box.w, box.h]

    @mcp.tool()
    def screenshot() -> dict:
        """截图，返回结构化信息而非原图：OCR全文+坐标（box为[x,y,w,h]）"""
        image = controller.post_screencap().wait().get()

        ocr_texts: list[dict[str, Any]] = []
        job = tasker.post_recognition(JRecognitionType.OCR, JOCR(threshold=0.0), image)
        detail = job.wait().get()
        if detail is not None and detail.nodes:
            reco = detail.nodes[0].recognition
            if reco is not None:
                for r in reco.all_results:
                    text = getattr(r, "text", None)
                    if text is not None:
                        ocr_texts.append(
                            {"text": text, "box": _box_to_list(r.box)}
                        )

        sensitive_hits = safety_guard.register_screenshot(ocr_texts)
        node_id, visited_count, is_novel = memory.visit([t["text"] for t in ocr_texts])

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
        result: dict[str, Any] = {
            "ocr_texts": ocr_texts,
            "image_ref": image_ref,
            "is_novel": is_novel,
            "visited_count": visited_count,
            "known_actions": known_actions,
            "known_ineffective": known_ineffective,
        }
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
        """点击坐标，坐标必须落在最近一次screenshot()返回的OCR候选框范围内"""
        safety_guard.validate_click(x, y)
        controller.post_click(x, y).wait()
        if state["current_node_id"] is not None:
            memory.record_action(
                state["current_node_id"], "click", [x, y], trigger_text=_trigger_text_at(x, y)
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

    ANDROID_KEYCODE_BACK = 4

    @mcp.tool()
    def press_back() -> str:
        """发送安卓系统返回键，用于关闭弹窗/浮层/返回上一级界面，不需要猜坐标"""
        controller.post_click_key(ANDROID_KEYCODE_BACK).wait()
        if state["current_node_id"] is not None:
            memory.record_action(state["current_node_id"], "press_back", None)
        return "pressed_back"

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
        action_type = action["type"]
        coords = action["coords"]
        if action_type == "press_back":
            controller.post_click_key(ANDROID_KEYCODE_BACK).wait()
        elif action_type == "swipe":
            safety_guard.validate_click_on_image(coords[0], coords[1])
            safety_guard.validate_click_on_image(coords[2], coords[3])
            controller.post_swipe(coords[0], coords[1], coords[2], coords[3], 300).wait()
        else:  # click / click_on_image：坐标是历史验证过的，仍然过一遍禁止点击区域
            safety_guard.validate_click_on_image(coords[0], coords[1])
            controller.post_click(coords[0], coords[1]).wait()

        memory.record_action(
            node_id, action_type, coords,
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
        print("用法: python maa_mcp_server.py <profile.yaml路径>", file=sys.stderr)
        sys.exit(1)

    server = build_server(sys.argv[1])
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
