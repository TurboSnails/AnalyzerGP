"""
查询扩写模块 — 混合策略（Hybrid）+ Query Metadata：

每条扩写 query 携带元数据，供 vector_store 按特征路由召回路径：

  RewriteQuery.type     → "original" | "semantic" | "keyword" | "api"
  RewriteQuery.weight   → 0.0~1.0，用于 Weighted RRF，原始问题权重最高
  RewriteQuery.routes   → 该 query 应送往哪几路召回（dense/hyde/bm25 子集）

路由策略：
  semantic  → ["dense", "hyde"]    概念性描述，向量语义匹配最好
  keyword   → ["bm25", "dense"]    错误类名/方法名，精确词面匹配为主
  api       → ["dense", "bm25"]    组件名+中文，两路互补
  original  → ["dense", "hyde", "bm25"]  原始问题走全路径

推理策略：
  简单查询  → Qwen2.5-1.5B 本地推理（~100-300ms，无网络）
  复杂查询  → MiniMax 远程推理（指代/极短/模糊，< 1μs 规则判断）
  降级链    → MiniMax 失败 → 本地 Qwen → 仍失败 → [RewriteQuery(原始)]
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_app1.core.config import OPENAI_API_KEY, QUERY_REWRITER_MODEL

if TYPE_CHECKING:
    import openai as _openai_t

logger = logging.getLogger("query_rewriter")

# ─── RewriteQuery 元数据 ──────────────────────────────────────────────────────

@dataclass
class RewriteQuery:
    """
    带路由元数据的检索查询单元。

    type    : 查询类型，决定最适合的召回路径
    weight  : Weighted RRF 权重（0~1），原始 query 始终 1.0
    routes  : 允许送入的召回路径子集，由 type 决定
    """
    text: str
    type: str               # "original" | "semantic" | "keyword" | "api"
    weight: float           # Weighted RRF 权重
    routes: list[str] = field(default_factory=lambda: ["dense", "hyde", "bm25"])

    def __str__(self) -> str:
        return f"[{self.type}|w={self.weight}|{'+'.join(self.routes)}] {self.text!r}"


# ─── Query 分类器 ─────────────────────────────────────────────────────────────
# 对 LLM 输出的每条纯文本 query 进行后处理分类，决定其 type/weight/routes

_CAMEL_OR_ERROR = re.compile(
    r'[a-z][A-Z]'                   # camelCase: onCreate, weakReference
    r'|[A-Z][a-zA-Z]+[A-Z]'         # UpperCamelCase: NullPointerException
    r'|\w+(Exception|Error)\b'       # 错误类名
    r'|\w+(Listener|Manager|Helper|Utils?|Factory)\b'  # Android 惯用后缀
)
_ANDROID_COMPONENTS = re.compile(
    r'\b(Activity|Fragment|Service|BroadcastReceiver|ContentProvider|'
    r'Handler|Looper|Thread|AsyncTask|Coroutine|ViewModel|LiveData|'
    r'RecyclerView|ViewHolder|Adapter|Intent|Bundle|Context|'
    r'Retrofit|OkHttp|Room|SQLite|SharedPreferences|'
    r'Hilt|Dagger|RxJava|Flow|LeakCanary|Glide|Picasso)\b'
)


def _classify(text: str, idx: int) -> tuple[str, float, list[str]]:
    """
    后处理分类：根据文本特征推断最适合的路由策略。

    idx=0 始终是原始 query，不分类，走全路径。
    """
    if idx == 0:
        return "original", 1.0, ["dense", "hyde", "bm25"]

    has_code_pattern  = bool(_CAMEL_OR_ERROR.search(text))
    has_component     = bool(_ANDROID_COMPONENTS.search(text))
    ascii_alpha_count = sum(1 for c in text if c.isascii() and c.isalpha())
    english_ratio     = ascii_alpha_count / max(len(text.replace(" ", "")), 1)

    # 英文为主 + 含代码模式 → keyword：BM25 精确词面命中为主，Dense 补充
    if english_ratio > 0.55 and has_code_pattern:
        return "keyword", 0.85, ["bm25", "dense"]

    # 含 Android 组件名 + 英文为主（无中文） → api：BM25 捕获 API 名，Dense 补充语义
    if has_component and english_ratio > 0.4:
        return "api", 0.75, ["bm25", "dense"]

    # 含 Android 组件名 + 中文混合 → api：Dense 语义理解 + BM25 组件名召回
    if has_component:
        return "api", 0.80, ["dense", "bm25"]

    # 默认：中文概念性描述 → semantic：Dense + HyDE 向量语义匹配
    return "semantic", 0.90, ["dense", "hyde"]


# ─── 共用 System Prompt ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是 Android 开发知识库的检索优化专家。

任务：根据【对话历史】和【当前问题】，生成 3~5 条适合向量检索的 query。

要求：
1. 第一条保留原始问题（原文不改）
2. 将口语/代词转换为 Android 技术术语（"闪退"→"crash NullPointerException"，"这个错误"→具体错误名）
3. 补全上下文指代（结合对话历史解析"这个"、"上面"等代词）
4. 生成英文技术关键词版本（便于匹配英文文档片段）
5. 每条 query 长度 ≤ 60 字，聚焦一个技术意图

输出格式（严格遵守）：
- 只输出一个 JSON 数组
- 数组的每个元素必须是字符串，不能是数组
- 不加任何解释、标签或 markdown

正确示例：
["Handler 内存泄漏怎么解决", "Android Handler 持有外部类引用内存泄漏", "Handler memory leak WeakReference fix", "LeakCanary 检测 Handler 泄漏"]

错误示例（禁止）：
[["Handler 内存泄漏怎么解决", "Android开发"], ["memory leak", "WeakReference"]]
"""

# ─── 复杂度判断 ───────────────────────────────────────────────────────────────

_CONTEXT_REFS = frozenset([
    "这个", "那个", "这种", "那种", "这里", "那里", "这些", "那些",
    "上面", "下面", "刚才", "之前", "前面", "该", "此",
])
_VAGUE_TERMS = frozenset([
    "怎么回事", "什么意思", "什么原因", "为什么会", "这是为什么", "啥原因",
])


def _is_complex(query: str, history: list) -> bool:
    q = query.strip()
    if len(q) < 8 and history:
        return True
    if history and any(ref in q for ref in _CONTEXT_REFS):
        return True
    if any(term in q for term in _VAGUE_TERMS):
        return True
    return False


# ─── 输出解析 ─────────────────────────────────────────────────────────────────

def _parse_output(original: str, raw: str) -> list[RewriteQuery]:
    """
    将模型原始输出解析为 list[RewriteQuery]。
    原始 query 强制置顶（idx=0，type=original，weight=1.0，全路由）。
    其余 query 由 _classify() 根据文本特征自动分类路由。
    """
    candidates: list[str] = []

    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list):
                        # 嵌套数组：取第一个元素作为 query
                        if item and isinstance(item[0], str):
                            candidates.append(item[0])
                    elif isinstance(item, str):
                        candidates.append(item)
    except Exception:
        pass

    if not candidates:
        candidates = [
            line.lstrip("-•·* \t0123456789.)").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    seen: set[str] = set()
    texts: list[str] = []

    def _add(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())[:80]
        if len(q) >= 3 and q not in seen:
            seen.add(q)
            texts.append(q)

    _add(original)
    for item in candidates:
        _add(item)

    if not texts:
        texts = [original]

    result = []
    for idx, text in enumerate(texts[:5]):
        qtype, weight, routes = _classify(text, idx)
        result.append(RewriteQuery(text=text, type=qtype, weight=weight, routes=routes))

    if len(result) > 1:
        logger.info(
            f"查询扩写 ({len(result)} 条):\n"
            + "\n".join(f"  {q}" for q in result)
        )
    return result


# ─── 本地路径：Qwen2.5-1.5B-Instruct ─────────────────────────────────────────

class QueryRewriterService:
    """Qwen2.5-1.5B-Instruct 本地推理服务，懒加载 + 线程安全。"""

    def __init__(self, model_path: str | None = None) -> None:
        self._path = model_path or QUERY_REWRITER_MODEL
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"加载 QueryRewriter 本地模型: {self._path}")
            if torch.cuda.is_available():
                self._device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                self._device = "mps"
                dtype = torch.float16
            else:
                self._device = "cpu"
                dtype = torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self._path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._path, dtype=dtype,
            ).to(self._device)
            logger.info(f"QueryRewriter 本地模型加载完成 (device={self._device})")

    def expand(self, query: str, history: list) -> list[RewriteQuery]:
        self._ensure_model()

        history_text = "\n".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:200]}"
            for m in history[-4:]
        )
        user_content = (f"【对话历史】\n{history_text}\n\n" if history_text else "") \
                     + f"【当前问题】\n{query}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        try:
            with self._lock:
                import torch
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                    )
                new_ids = output_ids[0][inputs.input_ids.shape[1]:]
                raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return _parse_output(query, raw)
        except Exception as e:
            logger.warning(f"[本地] 推理失败: {e}")
            return [RewriteQuery(text=query, type="original", weight=1.0,
                                 routes=["dense", "hyde", "bm25"])]


# ─── 远程路径：MiniMax ────────────────────────────────────────────────────────

_remote_client: "_openai_t.OpenAI | None" = None


def _get_remote_client():
    global _remote_client
    if _remote_client is None:
        import openai
        _remote_client = openai.OpenAI(
            base_url="https://api.minimaxi.com/v1",
            api_key=OPENAI_API_KEY,
        )
    return _remote_client


def _remote_rewrite(query: str, history: list) -> list[RewriteQuery]:
    history_text = "\n".join(
        f"{m.get('role', '')}: {str(m.get('content', ''))[:200]}"
        for m in history[-4:]
    )
    user_content = (f"【对话历史】\n{history_text}\n\n" if history_text else "") \
                 + f"【当前问题】\n{query}"

    try:
        resp = _get_remote_client().chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_output(query, raw)
    except Exception as e:
        logger.warning(f"[远程] MiniMax 扩写失败，降级本地 Qwen: {e}")
        return _get_service().expand(query, history)


# ─── 全局单例 ─────────────────────────────────────────────────────────────────

_service: QueryRewriterService | None = None


def _get_service() -> QueryRewriterService:
    global _service
    if _service is None:
        _service = QueryRewriterService()
    return _service


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def rewrite_queries(query: str, history: list | None = None) -> list[RewriteQuery]:
    """
    混合策略查询扩写主入口，返回带路由元数据的 RewriteQuery 列表。

    简单查询 → 本地 Qwen2.5-1.5B
    复杂查询（指代/极短/模糊）→ MiniMax 远程

    每条 RewriteQuery 携带 type/weight/routes，供 vector_store 按特征路由。
    失败时返回 [RewriteQuery(原始, original, 1.0, 全路由)]。
    """
    query = re.sub(r"\s+", " ", query.strip())[:120]
    if not query:
        return []

    history = history or []

    if _is_complex(query, history):
        logger.debug(f"复杂查询，走远程路径: {query!r}")
        return _remote_rewrite(query, history)
    else:
        return _get_service().expand(query, history)
