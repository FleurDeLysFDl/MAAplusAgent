"""意图→目标定位：对应 软件探索Agent三大功能模块设计.md 模块1.1。

三级查找顺序（跟文档execute_intent()一致）：
1. match_known_task：已经沉淀的技能库（core/skill_store.py管理的LearnedSkill，
   模块1.3"技能沉淀"负责往里面写，这里只负责查）里有没有直接对得上的
2. semantic_search_states：状态图（ExplorationMemory）里有没有语义上接近目标
   的已知节点，有就直接导航过去（复用core/exploration_memory.py的path_to）
3. 都没有：回退给goal_directed_probe（模块1.2，还没实现），这里只返回一个
   "需要探测"的信号，不在这个模块里实现探测本身

跟真实语义/embedding搜索不一样：这里的"语义匹配"用的是字符2-gram的Jaccard
重合度（_char_bigrams/_text_similarity），不是向量检索。中文没有天然的词
边界，项目里也没有引入分词依赖（jieba之类），拿core/exploration_memory.py
那套_stable_tokens/_similarity直接用在整句话上不合适——那套是给OCR每一条
独立的短文本框设计的，会把一整句话当成单独一个token，两句话只要不完全相等/
互为子串就相似度为0，实测对"去商城兑换东西"这种自然语言意图完全不work。
2-gram重合度是个简单、不需要额外依赖、还算够用的近似：意图和节点描述只要
共享"兑换""任务"这类2字词根，就能捕捉到部分语义关联。缺点跟_stable_tokens
那套一样是只能捕捉字面重合，"关闭充值"和"停止付费"这种同义不同词的意图
匹配不到，等接入LLM语义判断（比模块1.2的LLM候选排序更早一步）时再升级。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploration_memory import ExplorationMemory  # noqa: E402
from skill_store import LearnedSkill, SkillStore  # noqa: E402

# 语义搜索/技能匹配的最低相似度门槛：低于这个分数就当"没找到"，交给goal_directed_probe
# 兜底，而不是勉强导航到一个字面沾点边、实际风马牛不相及的节点/技能。2-gram Jaccard
# 天然比词级重合度低（两句话哪怕主题相关，2-gram交集占并集的比例也不会很高），
# 实测"去商城兑换东西"vs"商城界面...兑换中心...兑换各种道具"能到0.2左右，
# 门槛定得比这个略低，卡住明显不相关的意图
MATCH_THRESHOLD = 0.15


def _char_bigrams(text: str) -> set[str]:
    """字符2-gram集合，中文不需要分词也能算的近似语义重合度量。文本长度<2
    （比如单字意图）时退化成把整个文本当一个"gram"，避免返回空集合。"""
    text = text.strip()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _text_similarity(a: str, b: str) -> float:
    """用overlap coefficient（交集/较小集合的大小）而不是Jaccard（交集/并集）：
    意图通常很短，节点描述/技能样本通常长得多（describe_node()鼓励写清楚这个
    界面是干什么的），Jaccard分母是并集，会被长的那一边稀释得很惨——实测"去
    商城兑换东西"对上商城描述"商城界面，显示兑换中心...兑换各种道具"，共享
    "商城""兑换"两个2-gram，Jaccard只有0.045（分母被描述里其它无关的2-gram
    撑大了），但这两个2-gram已经占了意图自身信息量的1/3，overlap coefficient
    算出来是0.33，更能反映"这条描述确实提到了意图关心的东西"。
    """
    bigrams_a, bigrams_b = _char_bigrams(a), _char_bigrams(b)
    if not bigrams_a or not bigrams_b:
        return 0.0
    smaller = min(len(bigrams_a), len(bigrams_b))
    return len(bigrams_a & bigrams_b) / smaller if smaller else 0.0


def match_known_task(intent: str, store: SkillStore) -> LearnedSkill | None:
    """从已沉淀的技能库里找跟intent最像的一条，相似度够不上MATCH_THRESHOLD
    就当没有——技能库刚起步时基本是空的（模块1.3还没落地自动沉淀），这个函数
    大概率会一直返回None，这是预期行为，不是bug。
    """
    best: LearnedSkill | None = None
    best_score = 0.0
    for skill in store.list_all():
        for example in skill.intent_examples:
            score = _text_similarity(intent, example)
            if score > best_score:
                best_score = score
                best = skill
    return best if best_score >= MATCH_THRESHOLD else None


def semantic_search_states(
    intent: str, memory: ExplorationMemory, top_k: int = 3
) -> list[tuple[str, float]]:
    """从ExplorationMemory已知节点里找跟intent最像的几个，按describe_node()记录
    的description字段做相似度比较。没有description的节点（还没被Agent
    describe_node过）跳过——空描述参与比较只会白白拉低分数，不会命中。

    返回按分数降序的[(node_id, score), ...]，只保留达到MATCH_THRESHOLD的，
    最多top_k个；一个都没有就返回空列表，调用方应该转去goal_directed_probe。
    """
    scored = []
    for node_id, node in memory.nodes.items():
        description = node.get("description", "")
        if not description:
            continue
        score = _text_similarity(intent, description)
        if score >= MATCH_THRESHOLD:
            scored.append((node_id, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def rank_ocr_candidates_by_relevance(
    intent: str, ocr_texts: list[dict[str, Any]], tried_texts: set[str]
) -> list[dict[str, Any]]:
    """模块1.2"目标导向探测"的候选排序：文档原文是llm_rank_by_relevance（真正
    调LLM给每个候选打"离目标有多近"的分），这里用跟match_known_task/
    semantic_search_states同一套字符2-gram重合度顶替——一致的近似手段，一致的
    局限（见文件头注释），等接入真LLM判断时再换。

    过滤掉已经试过的候选（tried_texts，通常是当前节点get_actions()里记录过
    trigger_text的那些），按跟intent的相关性降序排列，只返回还没试过、真正
    "新" 的候选，调用方从头开始逐个尝试。
    """
    scored = [
        (item, _text_similarity(intent, item["text"]))
        for item in ocr_texts
        if item["text"] not in tried_texts
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _score in scored]


def route_intent(intent: str, game_id: str, *, storage_dir: str = "exploration_logs") -> dict[str, Any]:
    """execute_intent()的"定位"这一半：只判断该走哪条路、目标是谁，不负责真的
    执行（执行需要连真实设备，见core/execute_intent.py）。

    storage_dir默认"exploration_logs"（真实数据的位置），测试时传tmp_path
    隔离，不碰真实数据。

    返回三种之一：
      {"action": "run_known_task", "skill": LearnedSkill}
      {"action": "navigate_to_state", "node_id": str, "score": float}
      {"action": "goal_directed_probe"}  # 模块1.2，还没实现，调用方目前应该报告
                                          # "做不到"而不是真的去probe
    """
    store = SkillStore(game_id, storage_dir=storage_dir)
    skill = match_known_task(intent, store)
    if skill is not None:
        return {"action": "run_known_task", "skill": skill}

    memory = ExplorationMemory(game_id, storage_dir=storage_dir)
    candidates = semantic_search_states(intent, memory)
    if candidates:
        node_id, score = candidates[0]
        return {"action": "navigate_to_state", "node_id": node_id, "score": score}

    return {"action": "goal_directed_probe"}
