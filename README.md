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

如果用雷电模拟器，可以在`.env`里填`LDCONSOLE_PATH`（`ldconsole.exe`的路径）和
`LDPLAYER_INDEX`，`agent_runner.py`启动时会自动检查模拟器有没有在跑、没跑就拉起，
等adb连上、系统开机完成才继续，不用每次手动开模拟器再跑代码；不填这两个变量就
维持原样，自己管理模拟器/用真机。也可以单独跑这一步（比如只是想把模拟器开起来，
不马上跑agent）：

```
./.venv/Scripts/python core/emulator.py games/无期迷途/profile.yaml
```

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
HelloAgents的`ReActAgent`，然后开始自由探索。探索记忆按game_id分目录存在
`exploration_logs/<game_id>/nodes/`下，一个界面节点一个json文件，记录了这个界面的OCR特征、
以及探索时在这个界面上验证过坐标的操作（点了哪里、导向了哪个节点），重复运行不会互相污染。

再次遇到已经探索过的界面（`screenshot()`返回`known_actions`非空）时，LLM会优先调用
`replay_action`直接重放缓存坐标，不用重新截图分析或看图定位。如果在`.env`里配置了
`LLM_MODEL_ID_ROUTINE`，这一步还会自动切到那个更便宜的纯文本模型（这一步只需要从已知
选项里选一个，不需要视觉能力）；探索没见过的新界面时始终用`LLM_MODEL_ID`那个多模态模型。

## 下指令而不是自由探索

```
./.venv/Scripts/python core/agent_runner.py games/无期迷途/profile.yaml "帮我领取今日签到奖励"
```

带第三个参数就是执行用户指令，不带就是自由探索（默认行为不变）。两种模式复用同一套
MCP工具/探索记忆/known_actions重放/双模型路由，只是prompt里的"目标"不一样。

运行时终端会打印一个本地网页地址（默认`http://127.0.0.1:8765`），浏览器打开就能实时看到
LLM调用/工具调用/敏感界面拦截等日志（`core/log_broadcaster.py`，给`ReActAgent`自带的
TraceLogger包了一层SSE推送，不影响原有的`memory/traces/`落盘）。端口可以在profile.yaml里加
`dashboard: {host: ..., port: ...}`覆盖。

## 接入新游戏

复制`games/_template/`，按里面的README checklist填。验证标准：接入过程中`core/`目录
不应该有任何改动。
