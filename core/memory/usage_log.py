"""使用记录：按game_id记一份"什么时候、以什么模式、输入了什么、结果如何"的历史。

跟core/memory/exploration_memory.py的定位不同——那边存的是"这个游戏长什么样"的状态图，
这里存的是"我们怎么用这个Agent操作这个不同的app"的操作历史，两者都按game_id分
文件（同一份exploration_logs/<game_id>/目录下），但内容和用途都不一样，不合并
进同一份数据里，不同app（不同game_id）的使用记录天然互不干扰。

存成一个game_id一份jsonl（exploration_logs/<game_id>/usage_log.jsonl），追加写——
使用记录本质是不可变的历史事件，只增不改，不像ExplorationMemory的节点数据那样
需要反复改写/合并。

三个记录点（见各自文件里的调用）：
- core/agent/agent_runner.py：自由探索/指令执行两种ReActAgent会话，跑完记一条
- core/intent/execute_intent.py：结构化意图执行（已知技能重放/状态导航/目标探测），
  每种路径的成功/失败/被拒绝都各自记一条

build_usage_context()是给以后"把使用历史喂给LLM当context"这个场景准备的——
跟core/agent/react_agent_vision.py那套长短期记忆压缩是同一个思路：最近的记录保留
完整细节（最近的因果关系需要看清楚），更早的压缩成聚合统计（按mode/结果类型
分组计数，不逐条罗列），这样不管usage_log.jsonl积累了多少年历史，喂给LLM的
这段文本大小始终有上限，不会跟着使用次数无限增长。目前没有任何调用方把这段
context真的塞进prompt（还没有需要"知道这个游戏之前怎么被用过"的功能），先把
压缩逻辑准备好，等有这个需求时直接用。

用法（离线查看某个游戏的使用历史）：
    python core/memory/usage_log.py <game_id>
    python core/memory/usage_log.py <game_id> --context   # 看压缩后、可以直接喂给LLM的版本
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


def _result_category(result: str) -> str:
    """"error: xxx具体报错信息各不相同"这类result如果原样计数，分组会越滚越碎
    （每条不同的报错都算一类），压缩效果无从谈起。只取冒号前的那部分当分类
    （"error: xxx" -> "error"），跟record_usage()约定的result命名习惯配合——
    简短、以状态词开头，方便就近截断分类，不需要真的解析/理解报错内容。
    """
    return result.split(":", 1)[0].strip() or "unknown"


def build_usage_context(
    game_id: str, *, recent_n: int = 10, storage_dir: str = "exploration_logs"
) -> str:
    """压缩后的使用历史，长度不随记录条数无限增长，可以直接拼进LLM prompt。

    最近recent_n条保留完整细节（时间/模式/输入/结果），更早的全部记录只保留
    聚合统计（按mode分组次数、按结果类型分组次数、时间跨度）——跟
    core/agent/react_agent_vision.py的SHORT_TERM_STEPS是同一个"近详远略"思路，
    只是这里压缩的是使用历史，那边压缩的是单次会话内的对话消息。
    """
    entries = read_usage_log(game_id, storage_dir=storage_dir)
    if not entries:
        return f"（{game_id}还没有使用记录）"

    recent = entries[-recent_n:]
    older = entries[:-recent_n] if len(entries) > recent_n else []

    lines: list[str] = []
    if older:
        by_mode: dict[str, int] = {}
        by_result: dict[str, int] = {}
        for entry in older:
            mode = entry.get("mode", "unknown")
            by_mode[mode] = by_mode.get(mode, 0) + 1
            category = _result_category(entry.get("result", "unknown"))
            by_result[category] = by_result.get(category, 0) + 1
        lines.append(
            f"更早的{len(older)}次使用（{older[0]['ts']} ~ {older[-1]['ts']}）：\n"
            f"  按模式分组: {by_mode}\n"
            f"  按结果分组: {by_result}"
        )

    lines.append(f"最近{len(recent)}次使用的完整记录：")
    for entry in recent:
        detail = {k: v for k, v in entry.items() if k != "ts"}
        lines.append(f"  [{entry['ts']}] {detail}")

    return "\n".join(lines)


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python core/memory/usage_log.py <game_id>", file=sys.stderr)
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

    if "--context" in sys.argv[2:]:
        print("\n压缩后的LLM context版本：")
        print(build_usage_context(game_id))


if __name__ == "__main__":
    main()
