"""使用记录：按game_id记一份"什么时候、以什么模式、输入了什么、结果如何"的历史。

跟core/exploration_memory.py的定位不同——那边存的是"这个游戏长什么样"的状态图，
这里存的是"我们怎么用这个Agent操作这个不同的app"的操作历史，两者都按game_id分
文件（同一份exploration_logs/<game_id>/目录下），但内容和用途都不一样，不合并
进同一份数据里，不同app（不同game_id）的使用记录天然互不干扰。

存成一个game_id一份jsonl（exploration_logs/<game_id>/usage_log.jsonl），追加写——
使用记录本质是不可变的历史事件，只增不改，不像ExplorationMemory的节点数据那样
需要反复改写/合并。

三个记录点（见各自文件里的调用）：
- core/agent_runner.py：自由探索/指令执行两种ReActAgent会话，跑完记一条
- core/execute_intent.py：结构化意图执行（已知技能重放/状态导航/目标探测），
  每种路径的成功/失败/被拒绝都各自记一条

用法（离线查看某个游戏的使用历史）：
    python core/usage_log.py <game_id>
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _log_path(game_id: str, storage_dir: str) -> Path:
    return Path(storage_dir) / game_id / "usage_log.jsonl"


def record_usage(
    game_id: str,
    mode: str,
    input_text: str | None,
    result: str,
    *,
    storage_dir: str = "exploration_logs",
    extra: dict[str, Any] | None = None,
) -> None:
    """追加一条使用记录。mode是"explore"/"instruction"/"intent_execute"这几种
    之一，result是简短的结果描述（比如"success"/"timeout"/"error: xxx"），
    extra放一些可选的补充信息（比如目标节点id、技能名），不同mode的extra
    字段不强制统一格式，只是原样塞进这条记录。
    """
    path = _log_path(game_id, storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "input": input_text,
        "result": result,
    }
    if extra:
        entry.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_usage_log(game_id: str, *, storage_dir: str = "exploration_logs") -> list[dict[str, Any]]:
    """读出某个game_id的全部使用记录，按时间顺序（文件本身是追加写，天然有序）。
    文件不存在、或某一行不是合法json（比如写入中途被打断）都跳过，不抛异常——
    这是给人查看历史用的辅助读取，不需要对轻微的数据损坏零容忍。
    """
    path = _log_path(game_id, storage_dir)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize_usage(game_id: str, *, storage_dir: str = "exploration_logs") -> dict[str, Any]:
    """给人快速看"这个游戏之前都被怎么用过"：总次数、按mode分组次数、最近一次
    使用的完整记录。"""
    entries = read_usage_log(game_id, storage_dir=storage_dir)
    if not entries:
        return {"total": 0, "by_mode": {}, "last": None}
    by_mode: dict[str, int] = {}
    for entry in entries:
        mode = entry.get("mode", "unknown")
        by_mode[mode] = by_mode.get(mode, 0) + 1
    return {"total": len(entries), "by_mode": by_mode, "last": entries[-1]}


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python usage_log.py <game_id>", file=sys.stderr)
        sys.exit(1)

    game_id = sys.argv[1]
    summary = summarize_usage(game_id)
    print(f"总使用次数: {summary['total']}")
    print(f"按模式分组: {summary['by_mode']}")
    if summary["last"]:
        print(f"最近一次: {summary['last']}")

    if "--all" in sys.argv[2:]:
        print("\n完整历史：")
        for entry in read_usage_log(game_id):
            print(f"  {entry}")


if __name__ == "__main__":
    main()
