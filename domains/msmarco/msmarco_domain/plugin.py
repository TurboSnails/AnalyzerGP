"""MS MARCO 通用英文问答领域插件。"""
from __future__ import annotations

import json
from pathlib import Path

from rag_framework.domain.base import (
    CollectionNames,
    DomainPlugin,
    DomainPrompts,
    QueryRoute,
)

_QUESTION_WORDS = frozenset([
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
])

_CONTEXT_REFS = frozenset(["it", "its", "this", "that", "they", "them", "their"])


class MSMarcoDomainPlugin(DomainPlugin):
    """MS MARCO 通用英文问答领域插件。"""

    def __init__(self) -> None:
        self._base_dir = Path(__file__).parent
        self._system_prompt = self._load_prompt("system.txt")
        self._hyde_template = self._load_prompt("hyde.txt")

    @property
    def name(self) -> str:
        return "msmarco"

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def prompts(self) -> DomainPrompts:
        return DomainPrompts(system=self._system_prompt, hyde=self._hyde_template)

    def get_collection_names(self) -> CollectionNames:
        return CollectionNames(
            parent="knowledge_base",
            child="knowledge_base",
            hyde="knowledge_base_hyde",
        )

    def classify_query(self, query: str, history: list[dict]) -> QueryRoute:
        words = query.lower().split()
        first_word = words[0] if words else ""

        # Short keyword query → BM25-heavy (e.g. "corporation definition")
        if len(words) <= 3:
            return QueryRoute(text=query, type="keyword", weight=0.85,
                              routes=["bm25", "dense"])

        # Wh-question → semantic + HyDE (e.g. "what causes inflation?")
        if first_word in _QUESTION_WORDS:
            return QueryRoute(text=query, type="semantic", weight=0.90,
                              routes=["dense", "hyde"])

        # Default: balanced dense + BM25
        return QueryRoute(text=query, type="semantic", weight=0.85,
                          routes=["dense", "bm25"])

    def get_term_mapping(self) -> dict[str, str]:
        return {}

    def get_eval_dataset(self) -> list[dict]:
        eval_path = self._base_dir / "eval" / "benchmark.json"
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def rewrite_router_rules(self, query: str, history: list[dict]) -> int | None:
        # MS MARCO queries are clean and well-formed.
        # Only rewrite for pronouns / context references in multi-turn.
        if history and any(w in query.lower().split() for w in _CONTEXT_REFS):
            return 2
        return 0

    def fallback_response(self, reason: str = "low_confidence") -> str:
        templates = {
            "low_confidence": (
                "I couldn't find sufficiently relevant information in the knowledge base "
                "to answer this question confidently. Please try rephrasing your query."
            ),
            "no_results": (
                "No relevant passages were found for your query. "
                "Please try different keywords."
            ),
            "out_of_scope": (
                "This question is outside the scope of the current knowledge base."
            ),
        }
        return templates.get(reason, templates["out_of_scope"])

    def get_hyde_prompt(self, chunk: str) -> str:
        if "{chunk}" in self._hyde_template:
            return self._hyde_template.format(chunk=chunk)
        return self._hyde_template + f"\n\n{chunk}"

    def estimate_tokens(self, text: str) -> int:
        # English: ~1.3 tokens per word (BPE average)
        return int(len(text.split()) * 1.3)

    def _load_prompt(self, filename: str) -> str:
        path = self._base_dir / "prompts" / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
