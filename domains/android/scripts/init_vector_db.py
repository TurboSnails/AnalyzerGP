"""
Android 知识库初始化脚本 v1（单集合，已被 v2 取代）

保留作为轻量验证工具，仅构建 android_parent 单层索引。
生产环境请使用 init_vector_db_v2.py。

运行方式：
    uv run python domains/android/scripts/init_vector_db.py [--data-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "rag_framework"))
sys.path.insert(0, str(_ROOT / "domains" / "android"))

from android_domain.plugin import AndroidDomainPlugin
from rag_framework.core.config import get_settings
from rag_framework.core.logger import get_logger, setup_logging
from rag_framework.embedding.sentence_transformer import STEmbedder
from rag_framework.indexing.indexer import IndexConfig, VectorIndexer
from rag_framework.retrieval.dense import DenseStore

_logger = get_logger("android.init_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Android 知识库初始化 v1（单集合）")
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "ai_app1" / "data"),
        help="文档目录（默认：ai_app1/data/）",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    data_dir = Path(args.data_dir)
    files = [p for p in sorted(data_dir.iterdir()) if p.is_file()]
    if not files:
        _logger.warning(f"数据目录为空: {data_dir}")
        sys.exit(0)

    embedder = STEmbedder(
        model_path=settings.embed_model_path,
        device=settings.embed_device,
    )
    dense_store = DenseStore(
        chroma_path=settings.chroma_db_path,
        embedder=embedder,
    )
    domain = AndroidDomainPlugin()

    cfg = IndexConfig(
        chunk_size=300,
        overlap=60,
        enable_child=False,
        enable_hyde=False,
        enable_bm25=False,
    )
    indexer = VectorIndexer(
        domain=domain,
        embedder=embedder,
        dense_store=dense_store,
        config=cfg,
    )

    stats = indexer.index_files(files)
    _logger.info(f"完成: {stats}")


if __name__ == "__main__":
    main()
