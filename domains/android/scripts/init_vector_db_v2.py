"""
Android 知识库初始化脚本 v2（三层索引：parent / child / hyde）

使用 rag_framework.indexing.VectorIndexer 构建：
  android_parent  — 512字 parent chunks（LLM 上下文）
  android_child   — 128字 child chunks（高精度向量匹配，带 parent_id 回溯）
  android_hyde    — LLM 生成假设性问题（带 parent_id 回溯）

运行方式：
    uv run python domains/android/scripts/init_vector_db_v2.py [--data-dir PATH] [--reset]

数据目录默认：ai_app1/data/
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
sys.path.insert(0, str(_ROOT / "domains" / "android"))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from android_domain.plugin import AndroidDomainPlugin
from rag_framework.core.config import get_settings
from rag_framework.core.logger import get_logger, setup_logging
from rag_framework.embedding.sentence_transformer import STEmbedder
from rag_framework.indexing.indexer import IndexConfig, VectorIndexer
from rag_framework.llm.local_client import LocalLLMClient
from rag_framework.llm.openai_client import OpenAILLMClient
from rag_framework.retrieval.dense import DenseStore
from rag_framework.retrieval.sparse import BM25Store

_logger = get_logger("android.init_v2")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Android 知识库初始化 v2")
    # 默认从 ai_app1/data 读取源文档，与 RAGSettings 默认路径保持一致
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "ai_app1" / "data"),
        help="文档目录（默认：ai_app1/data/）",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="清空已有向量数据库再重建"
    )
    parser.add_argument(
        "--no-hyde", action="store_true",
        help="跳过 HyDE 问题生成（加快速度）"
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    files = [p for p in sorted(data_dir.iterdir()) if p.is_file()]
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

    # 初始化组件
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
    domain = AndroidDomainPlugin()

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
    indexer = VectorIndexer(
        domain=domain,
        embedder=embedder,
        dense_store=dense_store,
        sparse_store=sparse_store,
        llm=llm,
        config=cfg,
    )

    def _progress(done: int, total: int) -> None:
        _logger.info(f"进度: {done}/{total}")
        _check_memory(f"[{done} chunks]")

    stats = indexer.index_files(files, on_progress=_progress)

    _logger.info("=" * 50)
    _logger.info("索引构建完成")
    _logger.info(f"  文件数    : {stats.total_files}")
    _logger.info(f"  parent 数 : {stats.total_chunks}")
    _logger.info(f"  hyde 数   : {stats.hyde_generated}")
    if stats.errors:
        _logger.warning(f"  错误 ({len(stats.errors)}):")
        for e in stats.errors:
            _logger.warning(f"    {e}")


if __name__ == "__main__":
    main()
