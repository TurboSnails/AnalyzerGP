"""
RAG Framework 异常体系

按严重程度分层，便于调用方做精细化错误处理。
"""


class RAGError(Exception):
    """框架根异常。所有框架异常均继承此类。"""
    pass


# ─── 配置层 ───────────────────────────────────────────────────────────────────

class ConfigError(RAGError):
    """配置错误：环境变量缺失、路径无效、值类型错误等。"""
    pass


# ─── 模型加载层 ─────────────────────────────────────────────────────────────────

class ModelLoadError(RAGError):
    """模型加载失败：文件不存在、格式不支持、依赖缺失等。"""
    pass


class ModelNotFoundError(ModelLoadError):
    """模型路径或名称找不到。"""
    pass


# ─── LLM 层 ─────────────────────────────────────────────────────────────────────

class LLMError(RAGError):
    """LLM 调用异常基类。"""
    pass


class LLMTimeoutError(LLMError):
    """LLM 请求超时。"""
    pass


class LLMRateLimitError(LLMError):
    """LLM 速率限制。"""
    pass


class LLMContentFilterError(LLMError):
    """LLM 内容过滤拦截。"""
    pass


# ─── 检索层 ─────────────────────────────────────────────────────────────────────

class RetrievalError(RAGError):
    """检索异常基类。"""
    pass


class VectorStoreError(RetrievalError):
    """向量数据库操作失败。"""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Collection 不存在且无法自动创建。"""
    pass


class BM25Error(RetrievalError):
    """BM25 索引操作失败。"""
    pass


class BM25IndexEmptyError(BM25Error):
    """BM25 索引为空，未构建或构建失败。"""
    pass


class QueryRewriteError(RetrievalError):
    """查询改写失败。"""
    pass


# ─── Rerank 层 ──────────────────────────────────────────────────────────────────

class RerankError(RAGError):
    """精排异常。"""
    pass


class RerankFallbackError(RerankError):
    """CrossEncoder 不可用，已降级到规则排序。"""
    pass


# ─── Session 层 ─────────────────────────────────────────────────────────────────

class SessionError(RAGError):
    """会话异常。"""
    pass


class SessionNotFoundError(SessionError):
    """会话不存在。"""
    pass


# ─── 索引层 ─────────────────────────────────────────────────────────────────────

class IndexingError(RAGError):
    """索引构建异常。"""
    pass


class ChunkError(IndexingError):
    """分块异常。"""
    pass


class HyDEError(IndexingError):
    """HyDE 问题生成异常。"""
    pass


# ─── 领域层 ─────────────────────────────────────────────────────────────────────

class DomainError(RAGError):
    """领域插件异常。"""
    pass


class DomainNotFoundError(DomainError):
    """未找到指定领域插件。"""
    pass


class DomainConfigError(DomainError):
    """领域配置错误。"""
    pass
