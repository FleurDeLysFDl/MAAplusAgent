# 通用手游 Agent 项目开发文档

> 技术栈：**HelloAgents**（Datawhale `hello-agents` 框架，ReActAgent）作为决策大脑 + **MAA/MaaFramework** 封装成 **MCP Server** 作为手脚，对手游做自由探索（非预设脚本）。第一个落地目标游戏：**无期迷途**（参考 [MBCCtools](https://github.com/quietlysnow/MBCCtools) 的现成资源）。架构设计上做到游戏无关，方便后续接入其他游戏。

---

## 1. 总体架构

```
┌─────────────────────────────────┐
│  HelloAgents (ReActAgent)         │  决策大脑：Thought → Action → Observation 循环
│  ToolRegistry                     │
│    ├─ MCPTool → MAA-MCP-Server    │  通过MCP协议调用手脚
│    └─ CircuitBreaker              │  连续失败自动熔断，防死循环
└────────────────┬──────────────────┘
                 │ MCP (stdio/sse)
                 ▼
┌─────────────────────────────────┐
│  MAA-MCP-Server（自建，游戏无关）    │  用 mcp SDK (FastMCP) 包装MaaFramework能力
│  启动时加载指定游戏的 profile.yaml  │
└────────────────┬──────────────────┘
                 │ MaaFramework Python API
                 ▼
┌─────────────────────────────────┐
│  MaaFramework Tasker/Controller  │  实际控制模拟器/设备（adb）
│  （无期迷途可直接复用MBCCtools现成资源）│
└─────────────────────────────────┘
```

**分工原则**：MAA不做任何决策，只负责"看"（截图/OCR/模板匹配）和"动"（点击/滑动）。所有"下一步该干什么"的判断交给LLM。MAA-MCP-Server和Agent核心保持游戏无关，游戏相关的一切（包名、分辨率、OCR语言、已有任务、敏感词）都收进一个可插拔的`profile.yaml` + 资源目录，换游戏不改核心代码。

---

## 2. 目录结构

```
core/                                # 游戏无关，理论上永远不因换游戏而改动
├── maa_mcp_server.py                 # MCP Server主程序，启动参数指定profile
├── exploration_memory.py             # 探索记忆（按game_id分命名空间去重）
├── safety_guard.py                   # 坐标校验+敏感词拦截+熔断
└── agent_runner.py                   # HelloAgents装配代码

games/
├── 无期迷途/
│   ├── profile.yaml                  # 包名/分辨率/adb地址/OCR语言/探索提示语
│   ├── resource/pipeline/            # 直接复用MBCCtools现成的确定性任务资源
│   ├── interface.json
│   └── sensitive_keywords.yaml       # 本游戏专属敏感词补充
└── _template/                        # 以后接入新游戏，复制这个改名即可
    ├── profile.yaml.example
    └── README.md                     # 接入checklist

exploration_logs/                     # 每个游戏一份探索日志（界面指纹+截图+OCR）
```

---

## 3. MAA-MCP-Server 设计

### 3.1 启动时绑定游戏Profile

```python
# maa_mcp_server.py
import sys, yaml
from mcp.server.fastmcp import FastMCP

profile = yaml.safe_load(open(sys.argv[1]))  # 如 games/无期迷途/profile.yaml

controller = AdbController(profile["adb"]["address"])
resource = Resource()
resource.post_bundle(profile["resource_path"]).wait()
tasker = Tasker()
tasker.bind(controller, resource)

mcp = FastMCP(f"maa-mcp-{profile['game_id']}")
```
每个游戏对应一个独立MCP Server进程实例，adb连接/Resource绑定天然隔离，不会有状态残留问题。

### 3.2 工具清单（两层暴露）

**原子层（自由探索用）**：
```python
@mcp.tool()
def screenshot() -> dict:
    """截图，返回结构化信息而非原图：OCR全文+坐标"""
    image = controller.post_screencap().wait().get()
    ocr_results = resource.run_recognition("OCR", image)
    return {
        "ocr_texts": [{"text": r.text, "box": r.box} for r in ocr_results],
        "image_ref": save_temp_and_get_id(image)
    }

@mcp.tool()
def get_raw_image(image_ref: str) -> str:
    """结构化信息不够用时，取原图base64给多模态LLM直接看"""
    ...

@mcp.tool()
def click(x: int, y: int) -> str:
    controller.post_click(x, y).wait()
    return "clicked"

@mcp.tool()
def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
    controller.post_swipe(x1, y1, x2, y2, duration_ms).wait()
    return "swiped"
```

**任务层（复用MBCCtools现成资产）**：
```python
@mcp.tool()
def run_known_task(task_name: str) -> dict:
    """识别出当前是已知确定性流程（比如每日签到）时直接调用，不用摸索"""
    result = tasker.post_task(task_name).wait()
    return {"success": result.succeeded, "task": task_name}

@mcp.tool()
def list_known_tasks() -> list[str]:
    return list(interface_json["task"].keys())
```

**双层设计的意义**：纯自由探索成本高、效率低。凡是MBCCtools已经写好的确定性流程，LLM应优先识别"这是已知流程"直接调`run_known_task`；只有真正没见过的界面才逐步摸索点击。探索和现成自动化资产互补，不是推倒重来。

### 3.3 为什么结构化返回优先于原始截图

- 省token/省钱：多模态调用比纯文本贵很多，每步都传图成本不可控。
- 坐标grounding可靠：点击坐标必须引用OCR识别出的真实bbox，而不是LLM在图上估算像素位置。
- `get_raw_image`兜底：没有文字的图标类按钮，OCR识别不出来时才请求原图。

---

## 4. profile.yaml：一个游戏需要声明什么

```yaml
game_id: wuqimitu
display_name: 无期迷途
package_name: com.dragon.mituos
resolution: [1280, 720]
language: zh_cn
ocr_model: zh_cn                      # 对应 MaaCommonAssets 模型
adb:
  address: 127.0.0.1:5555
resource_path: games/无期迷途/resource
interface_path: games/无期迷途/interface.json
exploration_goal_hint: >
  策略养成类手游，主要界面包括城建、派遣、探索地图。
  探索时优先关注新出现的建筑/功能入口。
```
`exploration_goal_hint`把游戏类型特性注入通用Prompt模板，不用给每个游戏写一套独立Prompt。

---

## 5. HelloAgents 侧实现

### 5.1 装配代码

```python
from hello_agents import ReActAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools.protocols import MCPTool
from hello_agents.tools.circuit_breaker import CircuitBreaker

llm = HelloAgentsLLM()  # 建议选有基础视觉能力的模型，配合get_raw_image兜底
registry = ToolRegistry()
registry.register_tool(MCPTool(server_url=f"stdio://./maa_mcp_server.py games/无期迷途/profile.yaml"))
registry.register_tool(CircuitBreaker(max_failures=3, cooldown_s=60))

agent = ReActAgent("explorer", llm, tool_registry=registry)
agent.run(build_exploration_prompt(profile))
```

### 5.2 探索Prompt模板

```python
EXPLORATION_PROMPT_TEMPLATE = """
你正在自由探索手游【{display_name}】。
游戏特点：{exploration_goal_hint}

你的目标：
1. 尽量发现你还没见过的界面和功能
2. 遇到已知任务时优先调用 run_known_task，不要重复摸索
3. 记录每个新界面的功能和意义

当前已探索步数：{step_count}/{max_steps}
"""
```

### 5.3 探索记忆——避免反复横跳

```python
class ExplorationMemory:
    def __init__(self, game_id: str, storage_dir="exploration_logs/"):
        self.path = f"{storage_dir}/{game_id}.json"
        self.visited_states = self._load()

    def fingerprint(self, ocr_texts: list[str]) -> str:
        return hashlib.md5("".join(sorted(ocr_texts)).encode()).hexdigest()

    def is_novel(self, ocr_texts: list[str]) -> bool:
        fp = self.fingerprint(ocr_texts)
        if fp in self.visited_states:
            return False
        self.visited_states.add(fp)
        return True
```
每步把"这个界面你已经来过N次了"作为Observation的一部分喂给LLM，配合HelloAgents自带的HistoryManager做长程上下文管理，引导LLM往还没探索过的方向走。按`game_id`分文件存，避免不同游戏的探索记录互相污染。

---

## 6. 安全护栏

风险敞口比预设脚本自动化更大（LLM实时决策），护栏必须做在MCP Server层，不能只靠Prompt约束：

- **坐标白名单**：`click`/`swipe`坐标必须落在最近一次`screenshot()`返回的OCR候选框范围内，拒绝任意坐标。
- **敏感界面拦截**：`screenshot()`返回如果OCR命中敏感词，直接在工具返回里强提示"检测到敏感界面，任务终止"，不交给LLM自行判断。
```python
GLOBAL_SENSITIVE_KEYWORDS = ["支付", "实名认证", "绑定手机", "删除账号", "充值"]

def load_sensitive_keywords(profile: dict) -> list[str]:
    game_specific = yaml.safe_load(open(profile.get("sensitive_keywords_path", "/dev/null")) or {})
    return GLOBAL_SENSITIVE_KEYWORDS + game_specific.get("extra", [])
```
- **CircuitBreaker熔断**：HelloAgents自带，连续失败自动停止，避免无效循环消耗token。
- **步数/时间硬上限**：Agent循环层面强制最大步数，到点强制停止。

---

## 7. 新增游戏的接入checklist（验证"是否通用"的标准）

1. 复制`games/_template/`改名
2. 填`profile.yaml`
3. 有现成MaaFramework资源包（类似MBCCtools）就放进`resource/`；没有的话留空，纯自由探索也能跑，只是没有`run_known_task`可用
4. 按需补充`sensitive_keywords.yaml`
5. `python maa_mcp_server.py games/新游戏/profile.yaml`启动
6. HelloAgents侧连这个MCP地址开始探索

**判断标准**：接入过程中`core/`目录一行代码都不改，才算真正通用；如果需要改核心代码，说明某个游戏相关的东西之前被错误地写死在核心层了。

---

## 8. 开发顺序

1. **先写MAA-MCP-Server**，用MaaFramework Python API包装`screenshot`/`click`/`swipe`/`run_known_task`，脱离LLM用手写脚本先测通。
2. **HelloAgents最小闭环**：ReActAgent + MCPTool，给个简单目标（"截图告诉我看到了什么"）验证协议打通。
3. **坐标grounding校验**：MCP Server层加白名单逻辑，这是安全性核心，优先级高。
4. **探索记忆**：解决反复横跳问题，跑几十步session测试。
5. **接入run_known_task**：验证"已知流程直接调用、未知界面才摸索"能否真的降低步数和成本。
6. **安全护栏+CircuitBreaker压测**：故意让Agent走到敏感界面附近，确认能被正确拦截。
7. **抽象验证**：把无期迷途相关的硬编码内容全部搬进`profile.yaml`/`games/无期迷途/`，确认功能无倒退。
8. **（可选）第二个游戏接入测试**：用第7步的checklist接入另一个游戏，验证核心代码是否真的不用改。

---

## 9. 这套架构值得讲的点

- **LLM当"实时决策者"，MAA当"手和眼"**：分工清晰，用MCP协议解耦，而不是把识别/控制逻辑和决策逻辑写在一起。
- **结构化感知优先于原图**：解决了纯视觉Agent常见的坐标定位不准问题，是这个架构能可靠落地的关键设计判断。
- **双层工具（原子+已知任务）**：体现了"用确定性自动化处理已知场景、LLM只处理未知场景"的成本意识，不是无脑全部丢给LLM。
- **游戏无关的核心+可插拔Profile**：体现工程抽象能力，checklist本身就是验证抽象是否到位的测试标准。
