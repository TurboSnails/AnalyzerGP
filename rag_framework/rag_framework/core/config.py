"""
RAG Framework 统一配置（Pydantic Settings）

集中管理所有环境变量，提供类型校验、默认值、自动解析。
支持 .env 文件热重载（运行时修改需重启进程）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 将传统 .env 文件加载进 os.environ，保证 os.getenv("OPENAI_API_KEY") 等兼容逻辑可用
for _env_file in (".env", "ai_app1/.env"):
    if Path(_env_file).exists():
        load_dotenv(_env_file, override=False)


# ─── 环境文件预加载（兼容旧 OPENAI_API_KEY） ───────────────────────────────────

_repo_root = Path(__file__).parent.parent.parent.parent  # 项目根目录
for _env_file in (_repo_root / ".env", _repo_root / "ai_app1" / ".env"):
    if _env_file.exists():
        load_dotenv(_env_file, override=False)

# ─── 模型路径解析（原 config.py 中提取） ────────────────────────────────────────

def _resolve_weights_dir(*candidates: Path) -> Path | None:
    """返回第一个包含 pytorch_model.bin 或 model.safetensors 的目录。"""

    def has_weights(directory: Path) -> bool:
        return (directory / "pytorch_model.bin").is_file() or (
            directory / "model.safetensors"
        ).is_file()

    for c in candidates:
        if c.is_dir() and has_weights(c):
            return c
    return None


def _default_repo_root() -> Path:
    """推断仓库根目录。

    当 rag_framework 以 editable 或 site-packages 安装时均可正确推断。
    策略：沿 __file__ 向上查找包含 pyproject.toml 的目录；
          若该目录同时包含 ai_app1/ 子目录则确认为根目录，
          否则继续向上查找更大的根目录。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            if (parent / "ai_app1").is_dir():
                return parent
            # 当前目录有 pyproject.toml 但无 ai_app1（如 rag_framework 子包），
            # 继续向上查找包含 ai_app1 的更大根目录
            for grandparent in parent.parents:
                if (grandparent / "pyproject.toml").is_file() and (grandparent / "ai_app1").is_dir():
                    return grandparent
            return parent
    # 兜底：传统相对路径（开发模式）
    return current.parent.parent.parent.parent


def _resolve_bge_m3_path() -> str:
    repo = _default_repo_root()
    for base in (repo, repo / "ai_app1"):
        candidate = base / "models" / "bge-m3"
        if found := _resolve_weights_dir(candidate):
            return str(found)
    return str(repo / "models" / "bge-m3")


def _resolve_reranker_path() -> str:
    repo = _default_repo_root()
    for base in (repo, repo / "ai_app1"):
        candidate = base / "models" / "bge-reranker-base"
        if found := _resolve_weights_dir(candidate):
            return str(found)
    return "BAAI/bge-reranker-base"


def _resolve_rewriter_path() -> str:
    repo = _default_repo_root()
    for base in (repo, repo / "ai_app1"):
        candidate = base / "models" / "qwen2.5-1.5b-instruct"
        if found := _resolve_weights_dir(candidate):
            return str(found)
    return "Qwen/Qwen2.5-1.5B-Instruct"


def _resolve_llm_local_path() -> str:
    """返回本地 LLM 模型路径（优先查找已下载的 qwen2.5-1.5b-instruct）。"""
    repo = _default_repo_root()
    for base in (repo, repo / "ai_app1"):
        candidate = base / "models" / "qwen2.5-1.5b-instruct"
        if found := _resolve_weights_dir(candidate):
            return str(found)
    return str(repo / "models" / "qwen2.5-1.5b-instruct")


def _default_chroma_path() -> str:
    repo = _default_repo_root()
    # 优先使用已存在的 ai_app1/data/chroma_db
    legacy = repo / "ai_app1" / "data" / "chroma_db"
    if legacy.exists():
        return str(legacy)
    return str(repo / "pre" / "chroma_db")


def _default_bm25_path() -> str:
    return str(Path(_default_chroma_path()).parent / "tantivy_bm25")


# ─── 主配置类 ───────────────────────────────────────────────────────────────────

class RAGSettings(BaseSettings):
    """
    RAG Framework 统一配置。

    所有字段均可通过环境变量或 .env 文件覆盖。
    命名规范：前缀 RAG_，字段名大写，如 RAG_LLM_BACKEND=ollama
    """

    model_config = SettingsConfigDict(
        env_file=[".env", "ai_app1/.env"],
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",  # 忽略未知环境变量，兼容旧配置
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_backend: Literal["minimax", "ollama", "openai", "local"] = "local"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_max_tokens: int = 512
    llm_local_model_path: str = Field(default_factory=_resolve_llm_local_path)

    @field_validator("llm_base_url", mode="after")
    @classmethod
    def _resolve_llm_base_url(cls, v: str, info) -> str:
        if v:
            return v
        backend = info.data.get("llm_backend", "local")
        presets = {
            "minimax": "https://api.minimaxi.com/v1",
            "ollama": "http://127.0.0.1:11434/v1",
            "openai": "https://api.openai.com/v1",
            "local": "",
        }
        return presets.get(backend, presets["minimax"])

    @field_validator("llm_model", mode="after")
    @classmethod
    def _resolve_llm_model(cls, v: str, info) -> str:
        if v:
            return v
        backend = info.data.get("llm_backend", "local")
        presets = {
            "minimax": "MiniMax-M2.7",
            "ollama": "qwen2.5:1.5b-instruct-q4_K_M",
            "openai": "gpt-4o-mini",
            "local": "qwen2.5-1.5b-instruct",
        }
        return presets.get(backend, presets["minimax"])

    @field_validator("llm_api_key", mode="after")
    @classmethod
    def _resolve_llm_api_key(cls, v: str, info) -> str:
        if v:
            return v
        # 兼容旧 OPENAI_API_KEY
        if env_key := os.getenv("OPENAI_API_KEY"):
            return env_key
        backend = info.data.get("llm_backend", "local")
        if backend in ("ollama", "local"):
            return backend
        return ""

    # ── Embedding ────────────────────────────────────────────────────────────
    embed_model_path: str = Field(default_factory=_resolve_bge_m3_path)
    embed_device: str = "auto"
    embed_batch_size: int = 32
    embed_normalize: bool = True

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model_path: str = Field(default_factory=_resolve_reranker_path)
    reranker_batch_size: int = 32
    reranker_max_length: int = 512

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_db_path: str = Field(default_factory=_default_chroma_path)
    bm25_index_dir: str = Field(default_factory=_default_bm25_path)

    # ── Query Rewriter ───────────────────────────────────────────────────────
    rewriter_backend: Literal["auto", "ollama", "local"] = "auto"
    rewriter_model: str = Field(default_factory=_resolve_rewriter_path)
    rewriter_max_tokens: int = 128
    rewriter_cache_size: int = 512
    rewriter_use_remote_fallback: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct-q4_K_M"
    ollama_timeout: float = 60.0

    # ── Retrieval ────────────────────────────────────────────────────────────
    rrf_k: int = 60
    dense_query_k: int = 25
    dense_top_k: int = 10
    hyde_query_k: int = 15
    hyde_top_k: int = 5
    bm25_top_k: int = 10
    rerank_top_k: int = 3
    max_child_distance: float = 1.3
    max_distance_legacy: float = 1.2
    low_confidence_threshold: float = 0.30

    # ── Session ──────────────────────────────────────────────────────────────
    max_history: int = 4
    default_token_budget: int = 4096

    # ── Indexing ─────────────────────────────────────────────────────────────
    parent_chunk_size: int = 512
    parent_overlap: int = 100
    child_chunk_size: int = 128
    child_overlap: int = 25
    index_batch_size: int = 100

    # ── Domain ───────────────────────────────────────────────────────────────
    active_domain: str = "android"

    # ── Concurrency & Timeout ─────────────────────────────────────────────────
    llm_max_concurrent: int = 3              # LLM API 最大并发数（Semaphore 门控）
    retrieval_branch_timeout: float = 10.0   # 单路检索（Dense/BM25）超时秒数
    retrieval_rerank_timeout: float = 15.0   # Rerank 超时秒数

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    # ── 兼容旧配置（从 ai_app1/.env 读取） ────────────────────────────────────
    openai_api_key: str = ""  # 作为 llm_api_key 的 fallback

    @property
    def resolved_llm_api_key(self) -> str:
        return self.llm_api_key or self.openai_api_key or ""


@lru_cache(maxsize=1)
def get_settings() -> RAGSettings:
    """
    获取全局 Settings 单例（线程安全，缓存）。

    注意：若 .env 修改后需热重载，请调用 reload_settings()。
    """
    return RAGSettings()


def reload_settings() -> RAGSettings:
    """强制重新加载配置（用于测试或热重载场景）。"""
    get_settings.cache_clear()
    return get_settings()
