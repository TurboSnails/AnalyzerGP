"""
Wealth AI 知识库初始化脚本（三层索引：parent / child / hyde）

支持双域统一索引（wealth_parent / wealth_child / wealth_hyde）
以及分域独立索引（macro_econ_* / corp_earnings_*）。

运行方式：
    uv run python domains/wealth/scripts/init_wealth_db.py [--data-dir PATH] [--reset] [--no-hyde] [--split-domain]

数据目录默认：domains/wealth/data/
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

# 确保项目根在 PYTHONPATH
_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "rag_framework"))
sys.path.insert(0, str(_ROOT / "domains" / "wealth"))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from wealth_domain.plugin import WealthDomainPlugin
from rag_framework.core.config import get_settings
from rag_framework.core.logger import get_logger, setup_logging
from rag_framework.embedding.sentence_transformer import STEmbedder
from rag_framework.indexing.indexer import IndexConfig, VectorIndexer
from rag_framework.llm.local_client import LocalLLMClient
from rag_framework.llm.openai_client import OpenAILLMClient
from rag_framework.retrieval.dense import DenseStore
from rag_framework.retrieval.sparse import BM25Store

_logger = get_logger("wealth.init")

MEM_WARN_PCT = 75
MEM_GC_PCT = 85
MEM_ABORT_PCT = 92


def _check_memory(label: str = "") -> bool:
    if not _HAS_PSUTIL:
        return True
    pct = psutil.virtual_memory().percent
    if pct >= MEM_ABORT_PCT:
        _logger.error(f"内存 {pct:.1f}% 超过中止阈值，安全退出 {label}")
        return False
    if pct >= MEM_GC_PCT:
        _logger.warning(f"内存 {pct:.1f}% 超 GC 阈值，强制回收 {label}")
        gc.collect()
    elif pct >= MEM_WARN_PCT:
        _logger.info(f"内存 {pct:.1f}% 较高 {label}")
    return True


def _build_indexer(
    domain: WealthDomainPlugin,
    embedder: STEmbedder,
    dense_store: DenseStore,
    sparse_store: BM25Store,
    llm: LocalLLMClient | OpenAILLMClient,
    cfg: IndexConfig,
) -> VectorIndexer:
    """构造 VectorIndexer 实例。"""
    return VectorIndexer(
        domain=domain,
        embedder=embedder,
        dense_store=dense_store,
        sparse_store=sparse_store,
        llm=llm,
        config=cfg,
    )


def _index_single_domain(
    files: list[Path],
    indexer: VectorIndexer,
    domain_label: str,
) -> None:
    """为单个域构建索引并打印统计。"""

    def _progress(done: int, total: int) -> None:
        _logger.info(f"[{domain_label}] 进度: {done}/{total}")
        _check_memory(f"[{domain_label} {done} chunks]")

    stats = indexer.index_files(files, on_progress=_progress)

    _logger.info(f"[{domain_label}] 索引构建完成")
    _logger.info(f"  文件数    : {stats.total_files}")
    _logger.info(f"  parent 数 : {stats.total_chunks}")
    _logger.info(f"  hyde 数   : {stats.hyde_generated}")
    if stats.errors:
        _logger.warning(f"  错误 ({len(stats.errors)}):")
        for e in stats.errors:
            _logger.warning(f"    {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wealth AI 知识库初始化")
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "domains" / "wealth" / "data"),
        help="文档目录（默认：domains/wealth/data/）",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="清空已有向量数据库再重建",
    )
    parser.add_argument(
        "--no-hyde", action="store_true",
        help="跳过 HyDE 问题生成（加快速度）",
    )
    parser.add_argument(
        "--split-domain", action="store_true",
        help="按 macro/corp 分域构建独立索引（默认统一 wealth_*）",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    # 收集所有 .txt 文件（支持子目录）
    files = sorted(p for p in data_dir.rglob("*.txt") if p.is_file())
    if not files:
        _logger.warning(f"数据目录为空: {data_dir}")
        sys.exit(0)
    _logger.info(f"发现 {len(files)} 个源文件")

    if args.reset:
        db_path = Path(settings.chroma_db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
            _logger.info(f"已清空向量数据库: {db_path}")
        bm25_path = Path(settings.bm25_index_dir)
        if bm25_path.exists():
            shutil.rmtree(bm25_path)
            _logger.info(f"已清空 BM25 索引: {bm25_path}")

    # 初始化共享组件
    embedder = STEmbedder(
        model_path=settings.embed_model_path,
        device=settings.embed_device,
    )
    dense_store = DenseStore(
        chroma_path=settings.chroma_db_path,
        embedder=embedder,
    )
    sparse_store = BM25Store(
        index_dir=settings.bm25_index_dir,
        chroma_path=settings.chroma_db_path,
    )
    if settings.llm_backend == "local":
        llm = LocalLLMClient(
            model_path=settings.llm_local_model_path,
            max_tokens=settings.llm_max_tokens,
            max_concurrent=settings.llm_max_concurrent,
        )
    else:
        llm = OpenAILLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.resolved_llm_api_key,
            model=settings.llm_model,
            backend=settings.llm_backend,
            max_tokens=settings.llm_max_tokens,
            max_concurrent=settings.llm_max_concurrent,
        )

    cfg = IndexConfig(
        chunk_size=512,
        overlap=100,
        child_chunk_size=128,
        child_overlap=25,
        hyde_batch_size=4,
        enable_child=True,
        enable_hyde=not args.no_hyde,
        enable_bm25=True,
    )

    domain = WealthDomainPlugin()

    if args.split_domain:
        # ── 分域独立索引 ──────────────────────────────────────────────────────
        macro_files = [f for f in files if "macro" in f.name.lower()]
        corp_files = [f for f in files if "corp" in f.name.lower() or "earnings" in f.name.lower()]
        other_files = [f for f in files if f not in macro_files and f not in corp_files]

        if macro_files:
            # 临时 hack：让 domain.get_collection_names() 返回 macro 版本
            # 由于 VectorIndexer 直接调用 get_collection_names()，我们需要
            # 在索引前切换 domain 的默认 collection 配置
            _logger.info(f"构建宏观域索引: {len(macro_files)} 文件")
            # 通过 monkey-patch 临时切换
            original_names = domain.get_collection_names
            domain.get_collection_names = domain.get_macro_collection_names  # type: ignore[method-assign]
            _index_single_domain(
                macro_files,
                _build_indexer(domain, embedder, dense_store, sparse_store, llm, cfg),
                "macro",
            )
            domain.get_collection_names = original_names  # type: ignore[method-assign]

        if corp_files:
            _logger.info(f"构建财报域索引: {len(corp_files)} 文件")
            original_names = domain.get_collection_names
            domain.get_collection_names = domain.get_corp_collection_names  # type: ignore[method-assign]
            _index_single_domain(
                corp_files,
                _build_indexer(domain, embedder, dense_store, sparse_store, llm, cfg),
                "corp",
            )
            domain.get_collection_names = original_names  # type: ignore[method-assign]

        if other_files:
            _logger.info(f"构建统一域索引: {len(other_files)} 文件")
            _index_single_domain(
                other_files,
                _build_indexer(domain, embedder, dense_store, sparse_store, llm, cfg),
                "unified",
            )
    else:
        # ── 统一索引（默认）─────────────────────────────────────────────────────
        _index_single_domain(
            files,
            _build_indexer(domain, embedder, dense_store, sparse_store, llm, cfg),
            "wealth",
        )

    _logger.info("=" * 50)
    _logger.info("Wealth AI 知识库初始化完成")


if __name__ == "__main__":
    main()
