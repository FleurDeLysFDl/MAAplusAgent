# 软件探索 Agent：三大功能模块设计

> 基于已实现的**自愈生成器**（截图+检测结果→合法Pipeline节点→校验→落地）和已设计的**探索驱动层**（AppGraph状态图+候选操作枚举+回溯），补齐三个上层功能：0.空闲自主探索+进度估计 1.接受用户意图执行任务 2.高危操作确认harness。

---

## 0. 空闲自主探索 + 进度估计

### 0.1 调度：拿到包，空闲时自己跑

```python
class IdleExplorationScheduler:
    def __init__(self, app_graph: AppGraph, idle_threshold_s: int = 30):
        self.app_graph = app_graph
        self.idle_threshold_s = idle_threshold_s
        self.last_user_activity = time.time()

    def on_user_activity(self):
        self.last_user_activity = time.time()

    def tick(self):
        if time.time() - self.last_user_activity > self.idle_threshold_s:
            if not self.app_graph.is_fully_explored():
                explore_one_step(self.app_graph)   # 复用探索驱动层的单步逻辑
```
拿到一个新App包，只要没有用户意图任务在跑、且空闲超过阈值，就自动调用探索驱动层跑一步。用户随时可以打断（`on_user_activity`），探索让位给意图执行，状态图是持久化的，打断不丢进度。

### 0.2 进度估计

开放式探索没有明确的"总数"，不能简单算"完成了多少分之多少"，用两个互补的指标：

**指标①：前沿收敛度**（主指标，越接近1说明越接近探完）
```python
def frontier_progress(app_graph: AppGraph) -> float:
    tried = sum(len(s.tried_actions) for s in app_graph.states.values())
    untried = sum(len(enumerate_candidates(s.detections)) - len(s.tried_actions)
                  for s in app_graph.states.values())
    return tried / (tried + untried) if (tried + untried) > 0 else 0.0
```

**指标②：新状态发现率**（辅助判断是否"实质性"接近完成，而不是候选枚举本身有偏差）
```python
def discovery_rate(app_graph: AppGraph, window: int = 20) -> float:
    """最近window步里，发现新状态的比例。趋近0说明已经很久没找到新界面了"""
    recent = app_graph.step_log[-window:]
    new_state_hits = sum(1 for step in recent if step["was_new_state"])
    return new_state_hits / len(recent) if recent else 1.0
```

**综合展示**：
```python
def estimate_progress(app_graph: AppGraph) -> dict:
    return {
        "states_discovered": len(app_graph.states),
        "frontier_progress": frontier_progress(app_graph),
        "recent_discovery_rate": discovery_rate(app_graph),
        "verdict": "接近探完" if frontier_progress(app_graph) > 0.9 and discovery_rate(app_graph) < 0.1
                    else "探索中"
    }
```
两个指标一起看：前沿收敛度高但发现率也高，说明候选枚举本身漏检了（还有很多可交互元素没被识别出来，需要检查`enumerate_candidates`的启发式规则）；两个指标都低，才是真正意义上"这个App探得差不多了"。

---

## 1. 接受用户意图，执行任务

### 1.1 意图 → 目标定位

用户给一句自然语言意图（比如"帮我把每日任务都做完"），流程：

```python
def execute_intent(intent: str, app_graph: AppGraph):
    # 1. 先查已沉淀的"已知任务"库（探索/历史执行中学会的技能）
    known_task = match_known_task(intent, app_graph.learned_skills)
    if known_task:
        return run_known_task(known_task.name)

    # 2. 没有现成技能，定位状态图里语义上接近目标的节点
    target_candidates = semantic_search_states(intent, app_graph.states)
    if not target_candidates:
        return goal_directed_probe(intent, app_graph)  # 图里都没有，边探边做

    path = shortest_path(app_graph, current_state(), target_candidates[0])
    return replay_path(path)
```

### 1.2 图里没覆盖到目标怎么办：目标导向探测

如果状态图里找不到语义匹配的节点（说明这块区域之前空闲探索还没走到），把探索驱动层的"候选操作选择"从"随便选一个没试过的"换成"LLM按与目标意图的相关性排序"：

```python
def goal_directed_probe(intent: str, app_graph: AppGraph):
    current = get_current_state(app_graph)
    candidates = [c for c in enumerate_candidates(current.detections)
                  if c["id"] not in current.tried_actions]
    ranked = llm_rank_by_relevance(candidates, intent)  # LLM给每个候选打"离目标有多近"的分
    for candidate in ranked:
        patch = self_healing_generate(build_probe_context(current, candidate))  # 复用自愈生成器
        result = apply_and_execute(patch)
        if intent_satisfied(result, intent):
            return result
        if is_dead_end(result, intent):
            continue  # 换下一个候选
    return "未能在预算步数内完成意图"
```
这里**没有新造机制**——自愈生成器和探索驱动层原样复用，唯一变化是候选排序标准从"未探索优先"变成"目标相关性优先"，这样意图执行和空闲探索共用同一套底层能力，不是两套代码。

### 1.3 技能沉淀

意图执行成功走出来的路径，如果之后又被相同/相似意图触发过几次，就把这条路径固化成一个具名的`known_task`（类似之前讨论的`run_known_task`），下次同样意图直接查表调用，不用再走目标导向探测：
```python
class LearnedSkill:
    name: str                 # 从intent自动摘要出的技能名
    intent_examples: list[str]
    action_sequence: list[str]  # 固化的Pipeline节点序列
    success_count: int
```

---

## 2. 高危操作确认 Harness

### 2.1 从"静默拦截"升级为"主动询问"

之前自愈机制里敏感词命中是直接终止；现在改成**暂停+询问**，因为用户可能确实想执行这个操作（比如意图执行里用户本来就是想做一次付费操作）：

```python
class RiskGate:
    def __init__(self, sensitive_keywords: list[str]):
        self.sensitive_keywords = sensitive_keywords

    def assess(self, candidate_action: dict, context: dict) -> str:
        text = candidate_action.get("target_text", "")
        if any(kw in text for kw in self.sensitive_keywords):
            return "high_risk"
        if candidate_action["type"] == "click" and looks_irreversible(context):
            return "high_risk"   # 比如"确认""提交""删除"这类按钮语义
        return "safe"
```

### 2.2 触发确认的两种场景

**场景A：空闲自主探索中遇到高危候选**
```python
def explore_one_step(app_graph: AppGraph):
    candidate = pick_next_candidate(...)
    if risk_gate.assess(candidate, context) == "high_risk":
        mark_pending_confirmation(app_graph, candidate)   # 先跳过，标记待确认，不阻塞探索继续
        return
    # 正常执行...
```
空闲探索模式下，遇到高危操作**默认跳过不做**，标记进"待确认队列"，不打断自主探索的连续性（毕竟没有用户在场，不应该弹出等待）。等用户下次上线，把待确认队列展示出来，用户可以选择性放行。

**场景B：意图执行中遇到高危操作**
```python
def execute_intent(intent: str, app_graph: AppGraph):
    ...
    if risk_gate.assess(next_action, context) == "high_risk":
        confirmed = ask_user_confirmation(
            action_description=describe_action(next_action, context),
            reason="该操作可能涉及：" + risk_gate.matched_keyword
        )
        if not confirmed:
            return "已跳过高危操作，意图执行终止/降级"
    apply_and_execute(next_action)
```
这里必须**同步阻塞等待用户确认**，因为意图执行是用户主动发起的、用户此刻大概率在场，跟空闲探索的"跳过留着以后问"策略不同。

### 2.3 确认信息展示

询问时要给用户足够判断依据，不能只说"要不要继续"：
```python
def describe_action(action: dict, context: dict) -> str:
    return f"当前界面：{context['ocr_summary']}\n即将执行：{action['type']} 「{action['target_text']}」\n触发原因：命中高危关键词「{risk_gate.matched_keyword}」"
```

---

## 3. 三个模块的关系

```
IdleExplorationScheduler ──┐
                            ├──▶ 共享 AppGraph + 自愈生成器 + RiskGate
execute_intent(用户意图)  ──┘

RiskGate 在两条路径里都被调用，但响应策略不同：
  空闲探索遇高危 → 跳过+记待确认
  意图执行遇高危 → 阻塞+同步询问
```

三个功能不是三套独立系统，是同一套底层能力（状态图、自愈生成器、风险评估）在不同触发场景下的两种调度策略+一层风险交互界面。

---

## 4. 开发顺序

1. **进度估计指标**：先把`frontier_progress`/`discovery_rate`接到已有的探索日志上，跑一遍历史数据验证指标是否合理（比如人工看几个"应该算探完了"的case，指标是否也判断为高进度）。
2. **RiskGate + 场景A（空闲跳过+待确认队列）**：改动小，先上线，验证探索不会被高危操作卡住。
3. **场景B（意图执行同步询问）**：需要先有意图执行的最小闭环（1.1的`match_known_task`+`semantic_search_states`部分）才能测这个。
4. **目标导向探测（1.2）**：验证LLM候选排序是否真的比随机/顺序尝试更快找到目标，用几个具体意图做A/B对比（比如"打开设置"这种明确目标）。
5. **技能沉淀（1.3）**：等目标导向探测跑顺了再加，属于优化项，不阻塞前面的核心链路。
