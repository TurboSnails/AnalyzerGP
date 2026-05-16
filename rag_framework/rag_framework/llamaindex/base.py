"""
LlamaIndex Retriever 适配器。

将 LlamaIndex QueryEngine 包装为框架统一的 Retriever 接口，
使 ai_app4 可以像使用 HybridRetriever 一样使用 LlamaIndex 的检索能力。

策略：
  - response_mode="no_text"（默认）：仅返回 source_nodes 作为 RetrievedDoc 列表，
    由上层 LLM 节点自行合成回答，与现有 HybridRetriever 行为一致。
  - response_mode="compact" / "tree_summarize" / "refine"：返回 QueryEngine 合成后的
    文本作为单个 RetrievedDoc，适合需要 LlamaIndex 内置 Response Synthesis 的场景。
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from rag_framework.core.config import RAGSettings
from rag_framework.core.exceptions import RetrievalError
from rag_framework.core.factories import register_llamaindex_retriever
from rag_framework.core.logger import retrieval_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.retrieval.base import Retriever, RetrievalResult, RetrievedDoc

if TYPE_CHECKING:
    from rag_framework.embedding.base import Embedder
    from rag_framework.llamaindex.index_config import IndexDescription


class LlamaIndexRetriever(Retriever):
    """
    LlamaIndex QueryEngine 的 Retriever 接口适配器。

    依赖 LlamaIndex 在运行时的惰性加载：模块顶部不 import llama_index，
    仅在首次调用 _ensure_engine() 时加载，避免 ai_app1/2/3 启动时产生依赖。
    """

    def __init__(
        self,
        settings: RAGSettings,
        embedder: Embedder,
        index_description: IndexDescription,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._desc = index_description
        self._engine: Any | None = None
        self._index: Any | None = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str | QueryRoute | list[QueryRoute],
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        执行检索。

        将 QueryRoute 翻译为 LlamaIndex 查询参数，通过 asyncio.to_thread
        卸载 QueryEngine.query() 避免阻塞事件循环。
        """
        start = time.monotonic()
        self._ensure_engine()

        # 统一提取查询文本
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, QueryRoute):
            query_text = query.text
        elif isinstance(query, list):
            query_text = query[0].text if query else ""
        else:
            query_text = str(query)

        try:
            result = await asyncio.to_thread(
                self._engine.query,
                query_text,
            )
        except Exception as exc:
            retrieval_logger.error(f"LlamaIndex 检索异常: {exc}")
            raise RetrievalError(f"LlamaIndex retrieval failed: {exc}") from exc

        docs = self._parse_response(result, query_text)
        latency_ms = (time.monotonic() - start) * 1000
        retrieval_logger.info(
            f"LlamaIndex 检索完成: query={query_text[:30]!r}, "
            f"docs={len(docs)}, latency={latency_ms:.0f}ms"
        )
        return RetrievalResult(
            docs=docs,
            query=query_text,
            latency_ms=latency_ms,
            metadata={
                "index_type": self._desc.index_type,
                "response_mode": self._desc.response_mode,
            },
        )

    def warmup(self) -> None:
        """预热：加载索引和 QueryEngine，避免首请求延迟。"""
        self._ensure_engine()
        retrieval_logger.info("LlamaIndexRetriever 预热完成")

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _ensure_engine(self, top_k: int = 10) -> None:
        """惰性加载 LlamaIndex 索引和 QueryEngine。"""
        if self._engine is not None:
            return

        # 惰性导入，避免 ai_app1/2/3 加载重型依赖
        try:
            from llama_index.core import (
                Settings,
                VectorStoreIndex,
                SummaryIndex,
                KeywordTableIndex,
                TreeIndex,
                KnowledgeGraphIndex,
                StorageContext,
                load_index_from_storage,
            )
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:
            raise RetrievalError(
                "LlamaIndex 未安装，请执行 `pip install llama-index-core`"
            ) from exc

        # 配置全局 Settings（embedding + LLM）
        # 复用 rag_framework 的 Embedder，通过适配器桥接
        Settings.embed_model = _EmbedderAdapter(self._embedder)
        # LLM 不用于检索（除非启用 response synthesis），此处设为空避免误调用
        Settings.llm = None

        persist_dir = self._desc.persist_dir
        if persist_dir:
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                self._index = load_index_from_storage(storage_context)
                retrieval_logger.info(f"LlamaIndex 从持久化目录加载: {persist_dir}")
            except Exception:
                retrieval_logger.warning(
                    f"持久化目录加载失败，将重新构建索引: {persist_dir}"
                )
                self._index = None

        if self._index is None:
            self._index = self._build_index()
            if persist_dir:
                self._index.storage_context.persist(persist_dir=persist_dir)
                retrieval_logger.info(f"LlamaIndex 索引已持久化: {persist_dir}")

        similarity_top_k = self._desc.similarity_top_k or top_k
        self._engine = self._index.as_query_engine(
            response_mode=self._desc.response_mode,
            similarity_top_k=similarity_top_k,
        )

    def _build_index(self) -> Any:
        """根据 IndexDescription 构建新索引。"""
        from llama_index.core import (
            VectorStoreIndex,
            SummaryIndex,
            KeywordTableIndex,
            TreeIndex,
            KnowledgeGraphIndex,
            Document,
        )
        from llama_index.core.node_parser import SentenceSplitter

        documents: list[Any] = []
        for path in self._desc.doc_paths:
            from pathlib import Path
            p = Path(path)
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                documents.append(Document(text=text, metadata={"source": str(p)}))
            elif p.is_dir():
                for f in p.rglob("*.txt"):
                    text = f.read_text(encoding="utf-8")
                    documents.append(Document(text=text, metadata={"source": str(f)}))

        if not documents:
            retrieval_logger.warning("LlamaIndex 索引构建: 未找到任何文档")
            # 返回空 VectorStoreIndex 作为兜底
            return VectorStoreIndex.from_documents([])

        # 分块参数
        parser_kwargs = self._desc.node_parser or {}
        chunk_size = parser_kwargs.get("chunk_size", 512)
        chunk_overlap = parser_kwargs.get("chunk_overlap", 50)
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        idx_type = self._desc.index_type
        if idx_type == "vector":
            return VectorStoreIndex.from_documents(documents, node_parser=node_parser)
        if idx_type == "summary":
            return SummaryIndex.from_documents(documents, node_parser=node_parser)
        if idx_type == "keyword":
            return KeywordTableIndex.from_documents(documents, node_parser=node_parser)
        if idx_type == "tree":
            return TreeIndex.from_documents(documents, node_parser=node_parser)
        if idx_type == "kg":
            return KnowledgeGraphIndex.from_documents(documents, node_parser=node_parser)

        retrieval_logger.warning(f"未知索引类型 '{idx_type}'，回退到 vector")
        return VectorStoreIndex.from_documents(documents, node_parser=node_parser)

    def _parse_response(self, response: Any, query_text: str) -> list[RetrievedDoc]:
        """将 LlamaIndex Response 解析为 RetrievedDoc 列表。"""
        docs: list[RetrievedDoc] = []

        # 若启用 response synthesis，response.response 为合成文本
        if self._desc.response_mode != "no_text" and hasattr(response, "response"):
            text = str(response.response)
            if text:
                docs.append(
                    RetrievedDoc(
                        id="llamaindex_synthesized",
                        text=text,
                        score=1.0,
                        source="llamaindex_synthesis",
                        metadata={"query": query_text},
                    )
                )
                return docs

        # 默认：提取 source_nodes
        source_nodes = getattr(response, "source_nodes", [])
        for i, node in enumerate(source_nodes):
            node_id = getattr(node, "node_id", f"node_{i}")
            text = getattr(node, "text", "")
            score = getattr(node, "score", 0.0)
            metadata = getattr(node, "metadata", {})
            docs.append(
                RetrievedDoc(
                    id=node_id,
                    text=text,
                    score=score,
                    source="llamaindex",
                    metadata=metadata,
                )
            )
        return docs


# ── Embedder 适配器 ──────────────────────────────────────────────────────────

class _EmbedderAdapter:
    """
    将 rag_framework Embedder 桥接为 LlamaIndex BaseEmbedding。

    LlamaIndex 需要 get_text_embedding / get_query_embedding 接口，
    本适配器内部调用 rag_framework 的同步 encode() 方法。
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def get_text_embedding(self, text: str) -> list[float]:
        result = self._embedder.encode([text])
        return result[0] if result else []

    def get_query_embedding(self, query: str) -> list[float]:
        return self.get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """兼容 LlamaIndex 旧版接口。"""
        return self.get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        """兼容 LlamaIndex 旧版接口。"""
        return self.get_query_embedding(query)


# ── 自注册 ────────────────────────────────────────────────────────────────────
register_llamaindex_retriever("default", LlamaIndexRetriever)
