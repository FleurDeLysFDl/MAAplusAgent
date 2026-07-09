# 通用手游 Agent

架构和设计详见《通用手游Agent开发文档.md》。第一个落地游戏是无期迷途，资源来自
[MBCCtools](https://github.com/quietlysnow/MBCCtools)。

## 环境准备

```
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

复制`.env.example`为`.env`并填写LLM相关的key（`LLM_MODEL_ID`/`LLM_API_KEY`/`LLM_BASE_URL`）。

模拟器/真机需开启adb调试，且`games/无期迷途/profile.yaml`里的`adb.address`要能连上；
如果本机装了多个adb，建议在profile里显式填`adb.adb_path`，否则会用
`maa.toolkit.Toolkit.find_adb_devices()`自动探测。

## 单独测试MAA-MCP-Server

不经过LLM，先验证MaaFramework这一层能不能跑通：

```
./.venv/Scripts/python core/maa_mcp_server.py games/无期迷途/profile.yaml
```

进程会常驻并通过stdio等待MCP客户端连接（比如用MCP Inspector连上去手动调用
`screenshot`/`click`/`swipe`/`list_known_tasks`/`run_known_task`）。

## 跑自由探索Agent

```
./.venv/Scripts/python core/agent_runner.py games/无期迷途/profile.yaml
```

`agent_runner.py`会自动拉起`maa_mcp_server.py`子进程（通过stdio），把它暴露的工具注册进
HelloAgents的`ReActAgent`，然后开始自由探索。探索记忆存在`exploration_logs/<game_id>.json`里，
按game_id分文件，重复运行不会互相污染。

## 接入新游戏

复制`games/_template/`，按里面的README checklist填。验证标准：接入过程中`core/`目录
不应该有任何改动。
