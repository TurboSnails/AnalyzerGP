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

推理后端（REWRITER_BACKEND 环境变量）：
  ollama     → HTTP 调本地 Ollama（推荐，~0.5s，60-80 tok/s，跨平台）
  local      → transformers + mps Qwen2.5-1.5B（兜底，~3s，10-15 tok/s）
  默认 auto  → 优先 ollama，不可用则 fallback 到 local

性能优化：
  - LRU 缓存：相同 (query, history_hash) 直接返回，热 query <1ms
  - max_new_tokens=128：4 条 query 输出 ~100 token 足够，省 50% 推理时间
  - 默认禁用远程：USE_REMOTE_FALLBACK=False 时完全离线，无 MiniMax 调用
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

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


# ─── 性能开关 ─────────────────────────────────────────────────────────────────

USE_REMOTE_FALLBACK = False   # True: 本地失败兜底远程；False: 完全离线
MAX_NEW_TOKENS = 128          # 4 条 query 输出 ~100 token 够用（原 256）
REWRITE_CACHE_SIZE = 512      # LRU 缓存条数

# 推理后端选择：ollama / local / auto（默认 auto，先试 ollama 再 local）
REWRITER_BACKEND = os.getenv("REWRITER_BACKEND", "auto").lower()
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_REWRITER_MODEL", "qwen2.5:1.5b-instruct-q4_K_M")
OLLAMA_TIMEOUT   = float(os.getenv("OLLAMA_TIMEOUT", "60"))   # 冷启动加载模型可能 10-15s

# ─── 共用 System Prompt ───────────────────────────────────────────────────────
# 注意：Qwen2.5-1.5B 对格式示例非常敏感，必须保留正确/错误示例对照

_SYSTEM_PROMPT = """\
Android 检索助手。基于历史+问题，输出 3~4 条向量检索 query 的 JSON 数组。
规则：第1条原文；含1条英文术语；每条≤60字；不要 markdown 包裹。
示例：["Handler 内存泄漏", "Android Handler 持有外部类引用", "Handler memory leak WeakReference", "LeakCanary Handler 检测"]"""

# ─── Rewrite Router：三级分流策略 ─────────────────────────────────────────────
# 核心思想：80% query 不需要 LLM rewrite，把 LLM 调用率降到 ≤30%
#   Level 0 : 直接 original 走多路召回 (~0ms)        ← 简单 API/英文技术词
#   Level 1 : 规则同义扩展，无 LLM (~1ms)            ← 含已知技术术语
#   Level 2 : LLM rewrite (Ollama ~1.5s)             ← 指代/模糊/长自然语言/多跳

_CONTEXT_REFS = frozenset([
    "这个", "那个", "这种", "那种", "这里", "那里", "这些", "那些",
    "上面", "下面", "刚才", "之前", "前面", "该", "此", "它", "他们",
])
_VAGUE_TERMS = frozenset([
    "怎么回事", "什么意思", "什么原因", "为什么会", "这是为什么", "啥原因",
    "什么鬼", "啥情况", "搞不懂", "不明白",
])

# 中文术语 → 英文 keyword 映射，用于 Level 1 规则扩展（命中即追加一条 keyword query）
# 只放高频且映射明确的，避免噪声。
_ZH_TO_EN_TERMS: dict[str, str] = {
    "内存泄漏": "memory leak",
    "内存溢出": "OutOfMemoryError OOM",
    "卡顿":     "ANR jank lag",
    "崩溃":     "crash exception",
    "异步":     "async coroutine",
    "线程":     "thread executor",
    "回调":     "callback listener",
    "生命周期": "lifecycle",
    "重组":     "recomposition",
    "重绘":     "redraw invalidate",
    "缓存":     "cache",
    "网络请求": "http request retrofit okhttp",
    "依赖注入": "dependency injection hilt dagger",
    "数据库":   "database room sqlite",
    "权限":     "permission",
    "通知":     "notification",
    "广播":     "BroadcastReceiver",
    "意图":     "Intent",
    "服务":     "Service",
    "片段":     "Fragment",
    "适配器":   "Adapter",
    "列表":     "RecyclerView list",
    "布局":     "layout view",
    "动画":     "animation",
    "权重":     "weight LinearLayout",
}


def _route_level(query: str, history: list) -> int:
    """
    判断 query 应走哪一级 rewrite 路径。返回 0 / 1 / 2。

    Level 2（必须 LLM）：
      - 含代词且有 history（需要解析"这个/上面..."）
      - 含模糊词（"怎么回事"等）
      - 长自然语言（>= 25 字）：可能是多概念混合，LLM 拆解更准
      - query 极短 + 有 history（可能是追问）

    Level 1（规则扩展）：
      - 命中 _ZH_TO_EN_TERMS 至少 1 个中文术语
      - 中等长度（12~24 字）且含 Android 组件名

    Level 0（不 rewrite）：
      - 短英文 API 名（如 "Handler 内存泄漏" 已含明确 API，BM25 直接秒）
      - 短中文技术词（< 12 字）且无代词无模糊
      - 一切兜底
    """
    q = query.strip()
    if not q:
        return 0

    if any(term in q for term in _VAGUE_TERMS):
        return 2
    if history and any(ref in q for ref in _CONTEXT_REFS):
        return 2
    if len(q) >= 20:
        return 2
    if len(q) < 8 and history:
        return 2

    has_zh_term = any(t in q for t in _ZH_TO_EN_TERMS)
    if has_zh_term:
        return 1

    has_component = bool(_ANDROID_COMPONENTS.search(q))
    if 10 <= len(q) <= 19 and has_component:
        return 1

    return 0


def _level0_passthrough(query: str) -> list[RewriteQuery]:
    """Level 0：原样多路召回，不生成额外 query。"""
    return [RewriteQuery(text=query, type="original", weight=1.0,
                         routes=["dense", "hyde", "bm25"])]


def _level1_rule_rewrite(query: str) -> list[RewriteQuery]:
    """
    Level 1：基于词典的规则同义扩展，~1ms 完成。

    策略：
      - 原 query 保留为 original（全路由）
      - 命中的中文术语 → 生成 "原query + 英文keyword" 形式的 keyword query
        （走 BM25 + Dense，强化英文 API 词面命中）
      - 含 Android 组件名 → 追加纯英文组件名 query 走 BM25
    最多输出 3 条，避免无意义膨胀。
    """
    result: list[RewriteQuery] = [
        RewriteQuery(text=query, type="original", weight=1.0,
                     routes=["dense", "hyde", "bm25"])
    ]
    seen = {query}

    for zh, en in _ZH_TO_EN_TERMS.items():
        if zh in query:
            # 把中文术语替换成英文，保留 query 中的上下文
            new_q = query.replace(zh, en).strip()
            if new_q and new_q not in seen and len(new_q) <= 80:
                result.append(RewriteQuery(
                    text=new_q, type="keyword", weight=0.85,
                    routes=["bm25", "dense"],
                ))
                seen.add(new_q)
            if len(result) >= 3:
                break

    if len(result) < 3:
        components = _ANDROID_COMPONENTS.findall(query)
        if components:
            api_q = " ".join(dict.fromkeys(components))   # 去重保序
            if api_q not in seen and len(api_q) >= 3:
                result.append(RewriteQuery(
                    text=api_q, type="api", weight=0.75,
                    routes=["bm25", "dense"],
                ))
                seen.add(api_q)

    return result


# ─── 输出解析 ─────────────────────────────────────────────────────────────────

def _parse_output(original: str, raw: str) -> list[RewriteQuery]:
    """
    将模型原始输出解析为 list[RewriteQuery]。
    原始 query 强制置顶（idx=0，type=original，weight=1.0，全路由）。
    其余 query 由 _classify() 根据文本特征自动分类路由。
    """
    candidates: list[str] = []

    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    def _flatten(obj) -> None:
        if isinstance(obj, str):
            candidates.append(obj)
        elif isinstance(obj, list):
            for x in obj:
                _flatten(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                _flatten(v)

    try:
        m_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
        m_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
        chosen = m_arr or m_obj
        if chosen:
            data = json.loads(chosen.group())
            _flatten(data)
    except Exception:
        pass

    if not candidates:
        candidates = [
            line.lstrip("-•·* \t0123456789.)").strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]

    seen: set[str] = set()
    texts: list[str] = []

    def _add(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())
        q = q.strip("\"'`，。")[:80]
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


# ─── LRU 缓存 ─────────────────────────────────────────────────────────────────

_cache: "OrderedDict[str, list[RewriteQuery]]" = OrderedDict()
_cache_lock = threading.Lock()
_cache_stats = {"hit": 0, "miss": 0}


def _cache_key(query: str, history: list) -> str:
    """对 (query, 最近 4 轮 history) 做 hash，作为缓存 key。"""
    h_text = "|".join(
        f"{m.get('role','')}:{str(m.get('content',''))[:80]}"
        for m in (history or [])[-4:]
    )
    raw = f"{query}\x1f{h_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> "list[RewriteQuery] | None":
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            _cache_stats["hit"] += 1
            return list(_cache[key])
        _cache_stats["miss"] += 1
        return None


def _cache_put(key: str, value: "list[RewriteQuery]") -> None:
    with _cache_lock:
        _cache[key] = list(value)
        _cache.move_to_end(key)
        while len(_cache) > REWRITE_CACHE_SIZE:
            _cache.popitem(last=False)


def cache_stats() -> dict:
    """返回缓存命中统计，用于评测/调优。"""
    with _cache_lock:
        total = _cache_stats["hit"] + _cache_stats["miss"]
        return {
            "size": len(_cache),
            "hit": _cache_stats["hit"],
            "miss": _cache_stats["miss"],
            "hit_rate": round(_cache_stats["hit"] / total, 3) if total else 0.0,
        }


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()
        _cache_stats["hit"] = 0
        _cache_stats["miss"] = 0


# ─── Ollama 后端：HTTP 调本地 daemon ─────────────────────────────────────────

class OllamaRewriterService:
    """
    通过 Ollama HTTP API 调本地量化模型推理（推荐生产用）。

    优势：
      - Q4_K_M 量化 + Metal kernels，60-80 tok/s（vs transformers+mps 10-15 tok/s）
      - 模型常驻显存，无加载开销
      - HTTP 解耦，将来切 Linux GPU 服务器零改动
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        self._available: bool | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(
                        base_url=self._base_url,
                        timeout=self._timeout,
                    )
        return self._client

    def is_available(self) -> bool:
        """检查 ollama 服务是否就绪且目标模型已 pull。结果缓存避免每次请求都探测。"""
        if self._available is not None:
            return self._available
        try:
            r = self._get_client().get("/api/tags")
            r.raise_for_status()
            models = {m["name"] for m in r.json().get("models", [])}
            if self._model in models or any(self._model.startswith(m.split(":")[0]) for m in models):
                logger.info(f"[Ollama] 服务就绪，model={self._model}")
                self._available = True
            else:
                logger.warning(
                    f"[Ollama] 服务就绪但模型未 pull: {self._model}. "
                    f"已有: {sorted(models)}. 执行: ollama pull {self._model}"
                )
                self._available = False
        except Exception as e:
            logger.warning(f"[Ollama] 服务不可用 ({self._base_url}): {e}")
            self._available = False
        return self._available

    def preload(self) -> None:
        """
        预热：发送一次空 generate 让 Ollama 把模型加载进显存常驻。
        启动时调用一次，后续推理就不会有 10-15s 冷启动开销。
        """
        if not self.is_available():
            return
        try:
            t0 = time.perf_counter()
            r = self._get_client().post(
                "/api/generate",
                json={"model": self._model, "prompt": "", "keep_alive": "30m"},
                timeout=60,
            )
            r.raise_for_status()
            logger.info(f"[Ollama] 模型预热完成 {(time.perf_counter()-t0)*1000:.0f}ms")
        except Exception as e:
            logger.warning(f"[Ollama] 预热失败（不影响后续懒加载）: {e}")

    def expand(self, query: str, history: list) -> list[RewriteQuery]:
        history_text = "\n".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:200]}"
            for m in history[-4:]
        )
        user_content = (f"【对话历史】\n{history_text}\n\n" if history_text else "") \
                     + f"【当前问题】\n{query}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "stream": False,
            "keep_alive": "30m",   # 30m 内有请求即续命；超过则卸载，节省显存给其他软件
            "options": {
                "temperature": 0.0,
                "num_predict": MAX_NEW_TOKENS,
                "top_p": 1.0,
                "num_ctx": 1024,        # rewrite 任务用不到 2048，缩小 KV cache 加速 prefill
                "num_thread": 8,        # M 系列 P-core 充分利用
            },
        }

        try:
            t0 = time.perf_counter()
            r = self._get_client().post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            raw = (data.get("message", {}).get("content") or "").strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            eval_count = data.get("eval_count", 0)
            tok_per_sec = (eval_count / (data.get("eval_duration", 1) / 1e9)) if eval_count else 0
            logger.info(
                f"[Ollama] 推理完成 {elapsed_ms:.0f}ms, "
                f"output_tokens={eval_count}, {tok_per_sec:.0f} tok/s"
            )
            return _parse_output(query, raw)
        except Exception as e:
            logger.warning(f"[Ollama] 推理失败: {e}")
            self._available = False  # 标记不可用，下次自动降级
            return [RewriteQuery(text=query, type="original", weight=1.0,
                                 routes=["dense", "hyde", "bm25"])]


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
            import transformers
            transformers.logging.set_verbosity_error()  # 屏蔽 transformers 内部有缺陷的 debug log 调用
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
            t0 = time.perf_counter()
            with self._lock:
                import torch
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                new_ids = output_ids[0][inputs.input_ids.shape[1]:]
                raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[本地] Qwen推理完成 {elapsed_ms:.0f}ms, output_tokens={len(new_ids)}")
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

_local_service: QueryRewriterService | None = None
_ollama_service: OllamaRewriterService | None = None
_service_lock = threading.Lock()


def _get_local_service() -> QueryRewriterService:
    global _local_service
    if _local_service is None:
        with _service_lock:
            if _local_service is None:
                _local_service = QueryRewriterService()
    return _local_service


def _get_ollama_service() -> OllamaRewriterService:
    global _ollama_service
    if _ollama_service is None:
        with _service_lock:
            if _ollama_service is None:
                _ollama_service = OllamaRewriterService()
    return _ollama_service


def _pick_backend():
    """
    根据 REWRITER_BACKEND 选择推理后端。
    auto: ollama 可用则用 ollama，否则 fallback 到 transformers + mps。
    """
    if REWRITER_BACKEND == "ollama":
        return _get_ollama_service()
    if REWRITER_BACKEND == "local":
        return _get_local_service()
    ollama = _get_ollama_service()
    if ollama.is_available():
        return ollama
    return _get_local_service()


def preload() -> None:
    """
    在应用启动时调用一次，预热 query rewriter 模型常驻内存。
    auto/ollama 模式下预热 Ollama；local 模式下预热 transformers。
    """
    backend = _pick_backend()
    if isinstance(backend, OllamaRewriterService):
        backend.preload()
    else:
        # transformers 路径：触发 _ensure_model 加载
        backend._ensure_model()


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def rewrite_queries(query: str, history: list | None = None) -> list[RewriteQuery]:
    """
    Rewrite Router：按 query 复杂度分流到三级路径，降低 LLM 调用率。

    流程：
      0. LRU 缓存命中 → 直接返回 (<1ms)
      1. _route_level 判断 Level (0/1/2)
         Level 0 (~0ms)   : 原 query 直接多路召回（80% 简单 query 走这条）
         Level 1 (~1ms)   : 规则同义扩展（中文术语→英文 keyword）
         Level 2 (~1500ms): Ollama LLM rewrite（指代/模糊/长自然语言/多跳）
      2. 结果入缓存（包括 Level 0/1，相同 query 第二次更快）

    目标：将 LLM 调用率从 100% 降到 ≤30%，平均 TTFT 显著下降。
    """
    query = re.sub(r"\s+", " ", query.strip())[:120]
    if not query:
        return []

    history = history or []

    key = _cache_key(query, history)
    cached = _cache_get(key)
    if cached is not None:
        logger.info(f"查询扩写 [cache hit] ({len(cached)} 条): {query!r}")
        return cached

    level = _route_level(query, history)

    if level == 0:
        result = _level0_passthrough(query)
        logger.info(f"查询扩写 [L0 passthrough] 1 条: {query!r}")
        _cache_put(key, result)
        return result

    if level == 1:
        result = _level1_rule_rewrite(query)
        logger.info(
            f"查询扩写 [L1 规则] {len(result)} 条: "
            + " | ".join(f"{q.type}:{q.text}" for q in result)
        )
        _cache_put(key, result)
        return result

    t0 = time.perf_counter()
    backend = _pick_backend()
    result = backend.expand(query, history)
    llm_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"查询扩写 [L2 LLM] {len(result)} 条, 耗时 {llm_ms:.0f}ms: {query!r}")

    is_fallback_single = (len(result) == 1 and result[0].type == "original"
                          and result[0].text == query)

    if is_fallback_single and isinstance(backend, OllamaRewriterService):
        logger.info(f"Ollama 扩写失败，降级 transformers: {query!r}")
        result = _get_local_service().expand(query, history)
        is_fallback_single = (len(result) == 1 and result[0].type == "original"
                              and result[0].text == query)

    if is_fallback_single and USE_REMOTE_FALLBACK:
        logger.info(f"本地扩写仅返回原始 query，尝试 MiniMax 远程兜底: {query!r}")
        remote_result = _remote_rewrite(query, history)
        if len(remote_result) > 1:
            result = remote_result

    _cache_put(key, result)
    return result
