"""
AndroidDomainPlugin 实现

集中 Android 领域专用的所有逻辑：
- 系统提示词
- 查询分类器（含 Android 组件正则、术语映射）
- Collection 命名
- HyDE prompt
- 评测集加载
- 重写路由规则
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag_framework.core.logger import retrieval_logger
from rag_framework.core.registry import register_domain
from rag_framework.domain.base import (
    DomainPlugin,
    CollectionNames,
    QueryRoute,
    DomainPrompts,
)


class AndroidDomainPlugin(DomainPlugin):
    """Android 开发助手领域插件。"""

    def __init__(self) -> None:
        self._base_dir = Path(__file__).parent
        self._system_prompt = self._load_prompt("system.txt")
        self._hyde_template = self._load_prompt("hyde.txt")
        self._terms = self._load_terms()

    @property
    def name(self) -> str:
        return "android"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def prompts(self) -> DomainPrompts:
        return DomainPrompts(
            system=self._system_prompt,
            hyde=self._hyde_template,
        )

    def get_collection_names(self) -> CollectionNames:
        return CollectionNames(
            parent="android_parent",
            child="android_child",
            hyde="android_hyde",
        )

    def classify_query(self, query: str, history: list[dict]) -> QueryRoute:
        """
        Android 专用查询分类。

        根据文本特征推断最适合的路由策略。
        """
        has_code = self._has_code_pattern(query)
        has_component = self._has_android_component(query)
        ascii_alpha = sum(1 for c in query if c.isascii() and c.isalpha())
        en_ratio = ascii_alpha / max(len(query.replace(" ", "")), 1)

        if en_ratio > 0.55 and has_code:
            return QueryRoute(text=query, type="keyword", weight=0.85,
                              routes=["bm25", "dense"])
        if has_component and en_ratio > 0.4:
            return QueryRoute(text=query, type="api", weight=0.75,
                              routes=["bm25", "dense"])
        if has_component:
            return QueryRoute(text=query, type="api", weight=0.80,
                              routes=["dense", "bm25"])
        return QueryRoute(text=query, type="semantic", weight=0.90,
                          routes=["dense", "hyde"])

    def get_term_mapping(self) -> dict[str, str]:
        return self._terms.copy()

    def get_eval_dataset(self) -> list[dict]:
        eval_path = self._base_dir / "eval" / "benchmark.json"
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def rewrite_router_rules(self, query: str, history: list[dict]) -> int | None:
        """
        Android 专用 Rewrite Router。

        返回 rewrite level：
          0 = 不 rewrite（短技术词，BM25 直接命中）
          1 = 规则扩展（命中中文术语映射）
          2 = LLM rewrite（含代词/模糊词/长句）
        """
        # Level 2 条件
        if history and any(w in query for w in _CONTEXT_REFS):
            return 2
        if any(w in query for w in _VAGUE_TERMS):
            return 2
        if len(query) >= 25:
            return 2
        if len(query) <= 4 and history:
            return 2

        # Level 1 条件
        if any(term in query for term in self._terms):
            return 1
        if 12 <= len(query) <= 24 and self._has_android_component(query):
            return 1

        # Level 0：兜底
        return 0

    def fallback_response(self, reason: str = "low_confidence") -> str:
        templates = {
            "low_confidence": (
                "【知识库提示】本次问题在 Android 开发知识库中未找到强相关内容。"
                "请直接回复用户：『抱歉，这个问题超出了我当前 Android 知识库的覆盖范围，"
                "建议提供更多上下文或换个问法。』不要凭通用知识展开回答。"
            ),
            "no_results": (
                "【知识库提示】检索引擎未返回任何文档。请直接回复用户："
                "『抱歉，知识库目前没有相关资料，请稍后重试或换个问题。』"
            ),
            "out_of_scope": (
                "抱歉，这个问题超出了我当前 Android 知识库的覆盖范围。"
            ),
        }
        return templates.get(reason, templates["out_of_scope"])

    # ─── 内部辅助 ───────────────────────────────────────────────────────────────

    def _load_prompt(self, filename: str) -> str:
        path = self._base_dir / "prompts" / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def _load_terms(self) -> dict[str, str]:
        path = self._base_dir / "terms" / "zh_to_en.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _has_code_pattern(text: str) -> bool:
        patterns = [
            r'[a-z][A-Z]',
            r'[A-Z][a-zA-Z]+[A-Z]',
            r'\w+(Exception|Error)\b',
            r'\w+(Listener|Manager|Helper|Utils?|Factory)\b',
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _has_android_component(text: str) -> bool:
        components = (
            r'\b(Activity|Fragment|Service|BroadcastReceiver|ContentProvider|'
            r'Handler|Looper|Thread|AsyncTask|Coroutine|ViewModel|LiveData|'
            r'RecyclerView|ViewHolder|Adapter|Intent|Bundle|Context|'
            r'Retrofit|OkHttp|Room|SQLite|SharedPreferences|'
            r'Hilt|Dagger|RxJava|Flow|LeakCanary|Glide|Picasso)\b'
        )
        return bool(re.search(components, text))


# 上下文代词、模糊词（原 query_rewriter.py 中提取）
_CONTEXT_REFS = frozenset([
    "这个", "那个", "这种", "那种", "这里", "那里", "这些", "那些",
    "上面", "下面", "刚才", "之前", "前面", "该", "此", "它", "他们",
])
_VAGUE_TERMS = frozenset([
    "怎么回事", "什么意思", "什么原因", "为什么会", "这是为什么", "啥原因",
    "什么鬼", "啥情况", "搞不懂", "不明白",
])

# 模块导入时自动注册领域插件
register_domain(AndroidDomainPlugin)
