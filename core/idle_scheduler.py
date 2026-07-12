"""空闲自主探索调度器：对应 软件探索Agent三大功能模块设计.md 模块0.1。

跟文档原始设计有一处不得不做的简化：文档假设"探索驱动层"能一次只跑单步
（explore_one_step），空闲判定和执行在同一个tick循环里；本项目的探索驱动层
（core/agent_runner.py的ReActAgent循环）是"一次调用跑到Finish/max_steps"的
整段式设计，没有"跑一步就把控制权交还调用方"这个粒度，要做到真正的单步会
牵动ReActAgent的核心循环，代价太大。这里退一步：空闲超过阈值时启动一整段
探索（子进程，遵照profile.yaml自己的max_steps/min_new_nodes），而不是原子的
单步；用户活动信号一旦出现，立刻终止这段探索子进程，让位给用户——效果上仍然
满足"用户随时可以打断、不阻塞用户操作"这条核心要求，只是打断粒度是"一次
子进程"而不是"一步"。

调度依据两个信号：
1. 空闲判定：core/agent_runner.py.USER_ACTIVITY_MARKER这个标记文件的mtime——
   agent_runner.py带指令跑（意图执行）时会主动touch它，调度器只读不写
2. 是否"探得差不多了"：复用core/progress_estimate.py的frontier_progress+
   discovery_rate（模块0.2），两个指标都达标才不再自动探索，避免打满一次又
   一次没有意义的探索会话

用法（持续跑在前台，或者配合任务计划程序/systemd这类工具常驻）：
    python core/idle_scheduler.py <profile.yaml路径> [idle_threshold_s] [poll_interval_s]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_runner import USER_ACTIVITY_MARKER, load_profile  # noqa: E402
from exploration_memory import ExplorationMemory  # noqa: E402
from progress_estimate import discovery_rate, frontier_progress  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_RUNNER_SCRIPT = Path(__file__).resolve().parent / "agent_runner.py"
TRACE_DIR = PROJECT_ROOT / "memory" / "traces"

DEFAULT_IDLE_THRESHOLD_S = 30
DEFAULT_POLL_INTERVAL_S = 5


def seconds_since_last_activity() -> float:
    """USER_ACTIVITY_MARKER不存在，说明从来没记录过用户活动，当成"一直空闲"
    （返回一个很大的数，永远超过阈值），而不是抛异常或者当成"刚刚活动过"——
    调度器的默认姿态应该是"没人告诉我有用户在，那就去探索"。
    """
    if not USER_ACTIVITY_MARKER.exists():
        return float("inf")
    try:
        last_ts = float(USER_ACTIVITY_MARKER.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return float("inf")
    return time.time() - last_ts


def is_fully_explored(game_id: str, window: int = 20) -> bool:
    """两个指标（见core/progress_estimate.py）都达标才认为"探得差不多了"：
    frontier_progress>0.9且discovery_rate<0.1，任一指标不达标都继续探索。
    没有任何trace文件（从没跑过）时，直接认为没探完，不用等指标。
    """
    memory = ExplorationMemory(game_id)
    fp = frontier_progress(memory)
    traces = sorted(TRACE_DIR.glob("trace-*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        return False
    dr = discovery_rate(traces[-1], window=window)
    return fp > 0.9 and dr < 0.1


def run_scheduler(
    profile_path: str,
    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> None:
    profile = load_profile(profile_path)
    game_id = profile["game_id"]
    current_proc: subprocess.Popen | None = None

    print(f"🕐 空闲调度器启动：{game_id}，空闲阈值{idle_threshold_s}秒，每{poll_interval_s}秒检查一次")

    while True:
        idle_for = seconds_since_last_activity()

        if current_proc is not None:
            if idle_for < idle_threshold_s:
                print(f"👤 检测到用户活动，终止当前空闲探索（pid={current_proc.pid}），让位给用户")
                current_proc.terminate()
                current_proc.wait(timeout=10)
                current_proc = None
            elif current_proc.poll() is not None:
                print("✅ 这段空闲探索自然结束")
                current_proc = None

        elif idle_for >= idle_threshold_s:
            if is_fully_explored(game_id):
                print("🏁 已经探得差不多了（frontier_progress>0.9且discovery_rate<0.1），不再自动探索")
            else:
                print(f"💤 空闲{idle_for:.0f}秒，启动一段自由探索")
                current_proc = subprocess.Popen(
                    [sys.executable, str(AGENT_RUNNER_SCRIPT), profile_path],
                    cwd=str(PROJECT_ROOT),
                )

        time.sleep(poll_interval_s)


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print(
            "用法: python idle_scheduler.py <profile.yaml路径> [idle_threshold_s] [poll_interval_s]",
            file=sys.stderr,
        )
        sys.exit(1)

    profile_path = sys.argv[1]
    idle_threshold_s = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_IDLE_THRESHOLD_S
    poll_interval_s = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_POLL_INTERVAL_S

    try:
        run_scheduler(profile_path, idle_threshold_s, poll_interval_s)
    except KeyboardInterrupt:
        print("\n👋 调度器已停止")


if __name__ == "__main__":
    main()
