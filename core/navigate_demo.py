"""真实设备导航演示：拿现有探索记忆（core/exploration_memory.py）里已经验证过的
节点+操作图，实际驱动模拟器从当前画面走到指定的目标节点，边走边截图确认每一步
落地是否符合预期——跟core/maa_mcp_server.py里screenshot()的自动导航是同一套
ExplorationMemory.path_to()/get_actions()逻辑，只是单独拎出来跑、不需要起MCP
Server、也不需要LLM参与，纯粹展示"已知图能不能真把设备带到目标节点"。

复用core/maa_mcp_server.py里"连接设备+加载资源"那部分模块级函数（_load_profile/
_resolve_adb/_bind_resource），不复用build_server()内部的MCP工具包装。

用法：
    python core/navigate_demo.py <profile.yaml路径> <目标node_id>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2

from maa.controller import AdbController
from maa.define import LoggingLevelEnum
from maa.pipeline import JOCR, JRecognitionType
from maa.resource import Resource
from maa.tasker import Tasker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulator import ensure_emulator_ready  # noqa: E402
from exploration_memory import ExplorationMemory  # noqa: E402
from image_similarity import images_look_same  # noqa: E402
from maa_mcp_server import _bind_resource, _load_profile, _resolve_adb  # noqa: E402
from safety_guard import (  # noqa: E402
    SafetyGuard,
    load_click_forbidden_keywords,
    load_sensitive_keywords,
)

ANDROID_KEYCODE_BACK = 4
OCR_CONFIDENCE_THRESHOLD = 0.3


def _box_to_list(box: Any) -> list[int]:
    if isinstance(box, (list, tuple)):
        return [int(v) for v in box]
    return [box.x, box.y, box.w, box.h]


def _capture(tasker: Tasker, controller: AdbController) -> tuple[Any, list[dict[str, Any]]]:
    image = controller.post_screencap().wait().get()
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
    return image, ocr_texts


def _visit(memory: ExplorationMemory, image: Any, ocr_texts: list[dict[str, Any]]) -> tuple[str, int, bool]:
    def tiebreak(candidate_id: str) -> bool:
        path = memory.nodes_dir / f"{candidate_id}.png"
        if not path.exists():
            return False
        candidate = cv2.imread(str(path))
        return candidate is not None and images_look_same(candidate, image)

    return memory.visit(ocr_texts, tiebreak=tiebreak)


def _perform(controller: AdbController, safety_guard: SafetyGuard, action_type: str, coords: list[int] | None) -> None:
    """跟maa_mcp_server.py的_perform_action是同一套逻辑：重放一个已经验证过坐标
    的动作，仍然过一遍safety_guard的禁止点击区域校验"""
    if action_type == "press_back":
        controller.post_click_key(ANDROID_KEYCODE_BACK).wait()
    elif action_type == "swipe":
        safety_guard.validate_click_on_image(coords[0], coords[1])
        safety_guard.validate_click_on_image(coords[2], coords[3])
        controller.post_swipe(coords[0], coords[1], coords[2], coords[3], 300).wait()
    else:
        safety_guard.validate_click_on_image(coords[0], coords[1])
        controller.post_click(coords[0], coords[1]).wait()


def connect_device(profile: dict[str, Any]) -> tuple[Tasker, AdbController, SafetyGuard]:
    """拉起模拟器+连接设备+加载OCR资源，返回可以直接拿去截图/点击的三件套。

    core/navigate_demo.py和core/review_pending.py共用这套连接逻辑——两边都是
    "不起MCP Server、不需要LLM参与，直接用MaaFramework摸设备"的场景。
    """
    game_id = profile["game_id"]
    ensure_emulator_ready(profile)

    Tasker.set_stdout_level(LoggingLevelEnum.Off)
    Tasker.set_log_dir(Path(__file__).resolve().parent.parent / "exploration_logs" / "maa_debug" / game_id)

    adb_path, address = _resolve_adb(profile)
    controller = AdbController(adb_path=adb_path, address=address)
    if not controller.post_connection().wait().succeeded:
        raise RuntimeError(f"连接设备失败: {address}（adb_path={adb_path}）")

    resource = Resource()
    _bind_resource(resource, profile["resource_path"])

    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("Tasker绑定Resource/Controller失败")

    safety_guard = SafetyGuard(
        load_sensitive_keywords(profile),
        load_click_forbidden_keywords(profile),
    )
    return tasker, controller, safety_guard


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print("用法: python navigate_demo.py <profile.yaml路径> <目标node_id>", file=sys.stderr)
        sys.exit(1)

    profile_path, target_id = sys.argv[1], sys.argv[2]
    profile = _load_profile(profile_path)
    game_id = profile["game_id"]

    memory = ExplorationMemory(game_id)
    if target_id not in memory.nodes:
        print(f"❌ 目标节点不存在: {target_id}", file=sys.stderr)
        sys.exit(1)

    tasker, controller, safety_guard = connect_device(profile)

    print("📸 截图确认当前所在节点...")
    image, ocr_texts = _capture(tasker, controller)
    node_id, visited_count, is_novel = _visit(memory, image, ocr_texts)
    print(f"当前节点: {node_id}（{'新节点' if is_novel else f'第{visited_count}次访问'}）")

    if node_id == target_id:
        print("已经在目标节点，不需要导航")
        return

    path = memory.path_to(node_id, target_id)
    if path is None:
        print(f"❌ 已知图里没有从 {node_id} 到 {target_id} 的路径")
        sys.exit(1)

    print(f"🧭 导航路径（{len(path)}跳）: {node_id} -> {' -> '.join(path)}")

    current = node_id
    for step, next_id in enumerate(path, start=1):
        action = next(
            (a for a in memory.get_actions(current) if a.get("leads_to") == next_id),
            None,
        )
        if action is None:
            print(f"❌ 第{step}步：{current}上没有找到通向{next_id}的已知动作，导航中止")
            sys.exit(1)

        desc = action.get("trigger_text") or action["type"]
        print(f"第{step}步: {current} --[{action['type']} {desc}]--> {next_id}")
        _perform(controller, safety_guard, action["type"], action["coords"])

        memory.record_action(
            current, action["type"], action["coords"],
            trigger_text=action.get("trigger_text"),
            used_vision=action.get("used_vision", False),
        )
        # 真实LLM会话里，两次screenshot()之间天然隔着一次模型调用的往返延迟，
        # 界面转场/动画有充足时间播完；这里手动点击+截图之间没有这个天然延迟，
        # 点完立刻截图容易拍到转场播放到一半的画面，误判成"点了没反应"
        time.sleep(1.0)
        image, ocr_texts = _capture(tasker, controller)
        landed_id, _, _ = _visit(memory, image, ocr_texts)
        if landed_id != next_id:
            print(f"⚠️ 落地节点是{landed_id}，跟预期的{next_id}不符，导航中止（游戏状态可能已变化）")
            sys.exit(1)
        current = landed_id

    print(f"✅ 导航完成，已到达 {target_id}")


if __name__ == "__main__":
    main()
