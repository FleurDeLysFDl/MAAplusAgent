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
from maa.pipeline import JOCR, JRecognitionType
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402
from safety_guard import SafetyGuard, load_sensitive_keywords  # noqa: E402


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

    safety_guard = SafetyGuard(load_sensitive_keywords(profile))
    memory = ExplorationMemory(game_id)
    image_cache: dict[str, Any] = {}

    def _store_image(image: Any) -> str:
        ref = uuid.uuid4().hex
        image_cache[ref] = image
        if len(image_cache) > 20:
            image_cache.pop(next(iter(image_cache)))
        return ref

    mcp = FastMCP(f"maa-mcp-{game_id}")

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
                            {"text": text, "box": [r.box.x, r.box.y, r.box.w, r.box.h]}
                        )

        sensitive_hits = safety_guard.register_screenshot(ocr_texts)
        _, visited_count, is_novel = memory.visit([t["text"] for t in ocr_texts])

        result: dict[str, Any] = {
            "ocr_texts": ocr_texts,
            "image_ref": _store_image(image),
            "is_novel": is_novel,
            "visited_count": visited_count,
        }
        if sensitive_hits:
            result["sensitive_warning"] = (
                f"检测到敏感界面，命中关键词: {sensitive_hits}。任务终止，不要在此界面执行任何操作。"
            )
        return result

    @mcp.tool()
    def get_raw_image(image_ref: str) -> str:
        """结构化信息不够用时（比如没有文字的图标类按钮），取原图base64给多模态LLM直接看"""
        image = image_cache.get(image_ref)
        if image is None:
            raise ValueError(f"image_ref不存在或已过期: {image_ref}")
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @mcp.tool()
    def click(x: int, y: int) -> str:
        """点击坐标，坐标必须落在最近一次screenshot()返回的OCR候选框范围内"""
        safety_guard.validate_click(x, y)
        controller.post_click(x, y).wait()
        return "clicked"

    @mcp.tool()
    def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
        """滑动，起止坐标都必须落在最近一次screenshot()返回的OCR候选框范围内"""
        safety_guard.validate_swipe(x1, y1, x2, y2)
        controller.post_swipe(x1, y1, x2, y2, duration_ms).wait()
        return "swiped"

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
