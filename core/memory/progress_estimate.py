"""进度估计：给"这个游戏探得怎么样了"这个开放式问题一个量化答案。

对应 软件探索Agent三大功能模块设计.md 模块0.2。开放式探索没有"总数"，不能算
"完成了几分之几"，用两个互补指标一起看：

1. frontier_progress：已尝试动作数 / (已尝试 + 还没尝试的候选数)，跨所有已知
   节点汇总，越接近1说明摸得越透——是跨会话累积的"总体进度"
2. discovery_rate：某一次会话的trace日志里，最近window次screenshot()中发现
   新节点(is_novel)的比例，趋近0说明这次会话已经很久没找到新界面——是单次
   会话内"最近的发现节奏"，不跨会话累积

两个指标要一起看，不能只看一个：
- frontier_progress高、discovery_rate也高：矛盾信号，说明candidate枚举本身
  有偏差（untried_tokens这套OCR文字启发式漏检了很多真实可交互元素，比如纯
  图标按钮——它们从来不会计入untried_tokens，也就永远不会拉低frontier_progress，
  但探索确实还在发现新东西）
- 两个指标都低：真正意义上"探得差不多了"
- 两个指标都高：那次会话大概率已经在做无意义的重复尝试，该考虑喊停

frontier_progress读ExplorationMemory持久化的节点+操作图（跨会话累积，见
core/memory/exploration_memory.py）；discovery_rate读单次会话的trace-*.jsonl
（memory/traces/下，core/agent/agent_runner.py每次跑生成的那份，core/agent/log_broadcaster.py
实时转播的也是这份数据），只反映"这一段时间"，不需要额外新增日志格式。

用法（离线验证，跟已有trace文件对齐）：
    python core/memory/progress_estimate.py <game_id> <trace文件路径> [window]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.memory.exploration_memory import ExplorationMemory  # noqa: E402


def frontier_progress(memory: ExplorationMemory) -> float:
    """已尝试动作数 / (已尝试 + 还没尝试的候选)，跨所有已知节点汇总。

    "已尝试"用node["actions"]的条数（不管有没有效果，试过就算，跟
    get_ineffective_actions()的口径一致——ineffective也是"试过"的一种）；
    "还没尝试的候选"复用untried_tokens()的启发式（OCR稳定文字里还没被点击类
    操作覆盖的部分），跟has_frontier()/BFS自动导航用的是同一套判断口径，不是
    另起一套统计标准。
    """
    tried = sum(len(node.get("actions", [])) for node in memory.nodes.values())
    untried = sum(len(memory.untried_tokens(node_id)) for node_id in memory.nodes)
    total = tried + untried
    return tried / total if total else 0.0


def discovery_rate(trace_path: str | Path, window: int = 20) -> float:
    """某一次会话的trace文件里，最近window次screenshot()中发现新节点(is_novel)
    的比例。趋近0说明这次会话已经很久没找到新界面，可能是探得差不多了，也
    可能是卡在死路里出不来——配合frontier_progress一起看才能分清是哪种（低
    frontier_progress+低discovery_rate才是"卡住"，高frontier_progress+低
    discovery_rate才是"探完了"）。

    trace文件不存在/为空返回1.0（没有历史可看，不该被误判成"已经没有发现"）。
    """
    novel_flags: list[bool] = []
    path = Path(trace_path)
    if not path.exists():
        return 1.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "tool_result":
                continue
            payload = rec.get("payload", {})
            if payload.get("tool_name") != "screenshot":
                continue
            try:
                data = json.loads(payload["result"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            novel_flags.append(bool(data.get("is_novel")))
    recent = novel_flags[-window:]
    return sum(recent) / len(recent) if recent else 1.0


def estimate_progress(
    memory: ExplorationMemory, trace_path: str | Path, window: int = 20
) -> dict[str, Any]:
    fp = frontier_progress(memory)
    dr = discovery_rate(trace_path, window=window)
    verdict = "接近探完" if fp > 0.9 and dr < 0.1 else "探索中"
    return {
        "states_discovered": len(memory.nodes),
        "frontier_progress": round(fp, 4),
        "recent_discovery_rate": round(dr, 4),
        "verdict": verdict,
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print("用法: python core/memory/progress_estimate.py <game_id> <trace文件路径> [window]", file=sys.stderr)
        sys.exit(1)

    game_id, trace_path = sys.argv[1], sys.argv[2]
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    memory = ExplorationMemory(game_id)
    result = estimate_progress(memory, trace_path, window=window)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
