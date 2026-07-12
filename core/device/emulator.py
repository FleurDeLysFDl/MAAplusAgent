"""一键拉起本机模拟器 + 等它真正就绪，供agent_runner.py在连MCP Server之前调用。

只在.env配置了LDCONSOLE_PATH时才生效（目前只适配雷电模拟器/LDPlayer的
ldconsole.exe）；不配就认为是真机或者用户自己手动管理模拟器，这里整个跳过，
不影响这两种场景——它们的adb devices里设备本来就已经在，用不着"拉起"这一步。

换别的模拟器（MuMu/夜神/BlueStacks等）思路一样：命令行拉起实例 -> 等开机完成
-> adb connect，照着_launch_ldplayer/_wait_ready补一份对应实现即可，这里没有
做成插件式抽象，因为目前只有一台雷电模拟器需要支持，过度设计没意义。

拉起进程和"能用"是两码事：launch命令发出后进程立刻存在，但adbd监听、系统开机
流程都要再等一会——过早adb connect会连不上（reject/connection refused），过早
执行adb shell命令即使连上了也可能因为系统没启动完而拿不到正确结果。而且雷电
模拟器的`127.0.0.1:5555`这个TCP别名闲置一会儿还会自己掉线（同一台设备还有个
`emulator-5554`本地识别码撑着，掉的只是TCP别名），所以不能"connect一次成功就
只顾轮询boot_completed"，要在同一个重试循环里每轮都重新connect再查，才能扛住
中途掉线。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

READY_TIMEOUT = 240
POLL_INTERVAL = 3


def _run(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
    )


def _is_ldplayer_running(ldconsole_path: str, index: str) -> bool:
    result = _run([ldconsole_path, "isrunning", "--index", index])
    return result.stdout.strip() == "running"


def _launch_ldplayer(ldconsole_path: str, index: str) -> None:
    _run([ldconsole_path, "launch", "--index", index])


def _adb_connect_once(adb_path: str, address: str) -> bool:
    try:
        result = _run([adb_path, "connect", address], timeout=15)
        return "connected" in result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False


def _wait_ready(adb_path: str, address: str, timeout: int) -> None:
    """重试"connect+查boot_completed"直到设备真正可用

    connect和boot_completed查询要在同一轮里配对重试，不能"connect成功一次就
    只顾轮询boot_completed"：实测雷电模拟器同时用`127.0.0.1:5555`这个TCP别名
    和`emulator-5554`这个本地识别码代表同一台设备，TCP别名闲置一段时间会掉线
    （guess是没有keepalive），只做一次connect的话，boot_completed轮询轮到掉线
    那一刻就会一直报"device not found"直到耗尽timeout——哪怕设备其实早就开机
    完成了。每轮都重新connect一次（对已经连着的情况也就是拿到"already
    connected"，没有副作用）能让这条TCP别名掉线后自动重连回来。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _adb_connect_once(adb_path, address):
            try:
                result = _run(
                    [adb_path, "-s", address, "shell", "getprop", "sys.boot_completed"],
                    timeout=15,
                )
                if result.stdout.strip() == "1":
                    return
            except subprocess.TimeoutExpired:
                pass
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"等待模拟器({address})就绪超时({timeout}s)")


def ensure_emulator_ready(profile: dict[str, Any]) -> None:
    """profile.yaml里adb.address对应的设备真正可用后返回；没配LDCONSOLE_PATH就跳过。

    在agent_runner.py启动MCP Server子进程（子进程里AdbController会去连这个地址）
    之前调用，让"模拟器没开"这种情况在这里就给出清楚的中文提示并等它就绪，而不是
    让子进程那边报一个更晚、更难定位的连接失败。
    """
    ldconsole_path = os.getenv("LDCONSOLE_PATH")
    if not ldconsole_path:
        return
    if not Path(ldconsole_path).exists():
        raise RuntimeError(f"LDCONSOLE_PATH配置的路径不存在: {ldconsole_path}")

    index = os.getenv("LDPLAYER_INDEX", "0")
    address = profile["adb"]["address"]
    adb_path = profile.get("adb", {}).get("adb_path") or str(
        Path(ldconsole_path).with_name("adb.exe")
    )

    if not _is_ldplayer_running(ldconsole_path, index):
        print(f"📱 雷电模拟器实例{index}未运行，正在拉起...")
        _launch_ldplayer(ldconsole_path, index)
    else:
        print(f"📱 雷电模拟器实例{index}已在运行")

    print(f"🔌 等待adb连接{address}并确认开机完成...")
    _wait_ready(adb_path, address, READY_TIMEOUT)

    print("✅ 模拟器/adb已就绪")


def main() -> None:
    import sys

    import yaml
    from dotenv import load_dotenv

    # Windows控制台默认GBK编码，print里的emoji会直接UnicodeEncodeError崩掉——
    # 单独跑这个脚本时（不经过agent_runner.py）没有别人替我们reconfigure，得自己来
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python core/device/emulator.py <profile.yaml路径>", file=sys.stderr)
        sys.exit(1)

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    ensure_emulator_ready(profile)


if __name__ == "__main__":
    main()
