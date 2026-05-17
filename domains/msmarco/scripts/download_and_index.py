#!/usr/bin/env python3
"""
英文 QA 数据集下载 + 向量索引构建

使用 SQuAD v1.1（Stanford Question Answering Dataset，Wikipedia 段落 + QA），
完全公开免认证。与 MS MARCO 同属通用英文问答范畴，可等效测试 RAG 检索质量。

数据规模（validation split）：
  ~2,017 个唯一段落  × ~10,570 个问题 ≈ 真实 IR 检索场景

用法：
  cd fenxiCB
  uv run python domains/msmarco/scripts/download_and_index.py

注意：
  - ai_app1 运行时所有领域共用同一个 BM25 索引目录（由 RAG_BM25_INDEX_DIR 决定）。
  - 默认目录为 ai_app1/data/tantivy_bm25，与 android 领域索引共存，通过 domain 元数据隔离。
  - 无需手动覆盖 RAG_BM25_INDEX_DIR，保持与 android 一致即可。

可选参数：
  --reset       清空现有向量库后重建
  --no-hyde     跳过 HyDE 向量生成（加快速度）
  --split       train / validation（默认: validation）
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import shutil
import sys
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "uv.lock").exists():
            return parent
    raise RuntimeError("找不到项目根目录（需含 pyproject.toml + uv.lock）")


_ROOT = _find_project_root()
for _p in [str(_ROOT / "rag_framework"), str(_ROOT / "domains/msmarco")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msmarco_domain.plugin import MSMarcoDomainPlugin
from rag_framework.core.config import get_settings
from rag_framework.core.logger import get_logger, setup_logging
from rag_framework.embedding.sentence_transformer import STEmbedder
from rag_framework.indexing.hyde import generate_hyde_questions
from rag_framework.llm.local_client import LocalLLMClient
from rag_framework.llm.openai_client import OpenAILLMClient
from rag_framework.retrieval.dense import DenseStore
from rag_framework.retrieval.sparse import BM25Store

_logger = get_logger("msmarco.index")

_EMBED_BATCH = 256
_BM25_BATCH  = 500


def _passage_id(title: str, context: str) -> str:
    """生成段落 ID：title + context md5 前缀，保证唯一且可读。"""
    h = hashlib.md5(context.encode()).hexdigest()[:10]
    safe_title = title.replace(" ", "_")[:40]
    return f"{safe_title}_{h}"


# ─── 数据加载 ────────────────────────────────────────────────────────────────

def _load_unique_passages(split: str) -> tuple[list[str], list[str], list[dict]]:
    """
    从 SQuAD 加载所有唯一段落（按 context 去重）。

    Returns:
        ids, texts, metadatas
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _logger.error("请先安装 datasets 库: uv add datasets")
        sys.exit(1)

    _logger.info(f"加载 rajpurkar/squad [{split}] ...")
    ds = load_dataset("rajpurkar/squad", split=split)
    _logger.info(f"共 {len(ds)} 条 QA，开始去重段落...")

    seen: dict[str, str] = {}   # context → passage_id
    ids, texts, metadatas = [], [], []

    for row in ds:
        ctx  = row["context"].strip()
        key  = ctx  # 以完整 context 去重
        if key in seen:
            continue
        pid = _passage_id(row["title"], ctx)
        seen[key] = pid
        ids.append(pid)
        texts.append(ctx)
        metadatas.append({
            "parent_id": pid,
            "source": row["title"],
            "title": row["title"],
            "domain": "msmarco",
        })

    _logger.info(f"唯一段落数: {len(ids)}（原始 QA: {len(ds)} 条）")
    return ids, texts, metadatas


# ─── Dense 索引 ──────────────────────────────────────────────────────────────

def _index_dense(col, embedder, ids, texts, metadatas) -> None:
    total = len(ids)
    _logger.info(f"写入 DenseStore，共 {total} 条...")
    for start in range(0, total, _EMBED_BATCH):
        end = min(start + _EMBED_BATCH, total)
        embeddings = embedder.encode(texts[start:end])
        col.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        _logger.info(f"  Dense: {end}/{total}")
        if end % (5 * _EMBED_BATCH) == 0:
            gc.collect()


# ─── BM25 索引 ───────────────────────────────────────────────────────────────

def _index_bm25(sparse_store, ids, texts) -> None:
    total = len(ids)
    _logger.info(f"写入 BM25Store，共 {total} 条...")
    for start in range(0, total, _BM25_BATCH):
        end = min(start + _BM25_BATCH, total)
        sparse_store.add_documents(
            list(zip(ids[start:end], texts[start:end])), domain="msmarco"
        )
        _logger.info(f"  BM25: {end}/{total}")


# ─── HyDE 索引 ───────────────────────────────────────────────────────────────

async def _check_llm_connectivity(domain, llm) -> bool:
    """尝试一次 LLM 调用，返回 True 表示可用。"""
    try:
        prompt = domain.get_hyde_prompt("connectivity check")
        await llm.chat([{"role": "user", "content": prompt}])
        return True
    except Exception as e:
        _logger.warning(f"LLM 连接检查失败: {e}")
        return False


def _index_hyde(hyde_col, embedder, domain, llm, ids, texts, hyde_sample) -> None:
    sample_ids   = ids[:hyde_sample]
    sample_texts = texts[:hyde_sample]
    _logger.info(f"生成 HyDE 问题，共 {len(sample_ids)} 条段落...")

    # 在同一个事件循环中完成连通性检查和 HyDE 生成，
    # 避免跨循环复用同一个 OpenAILLMClient（其内部的 Semaphore 与 httpx.AsyncClient
    # 均绑定到创建时的事件循环，切换循环会导致 Connection error）。
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        reachable = loop.run_until_complete(_check_llm_connectivity(domain, llm))
        if not reachable:
            _logger.warning("LLM 服务不可达，跳过 HyDE 生成。请确认本地模型服务已启动，或使用 --no-hyde 跳过。")
            return

        questions = loop.run_until_complete(
            generate_hyde_questions(sample_texts, domain, llm, batch_size=4)
        )
    finally:
        loop.close()

    valid = [(q, pid) for q, pid in zip(questions, sample_ids) if q and q.strip()]
    if not valid:
        _logger.warning("HyDE 未生成任何问题，跳过写入")
        return

    hyde_texts, hyde_ids = zip(*valid)
    hyde_metas = [
        {"parent_id": pid, "source": f"hyde_{pid}", "domain": "msmarco"}
        for pid in hyde_ids
    ]

    _logger.info(f"写入 HyDE collection，共 {len(valid)} 条...")
    for start in range(0, len(valid), _EMBED_BATCH):
        end = min(start + _EMBED_BATCH, len(valid))
        embeddings = embedder.encode(list(hyde_texts[start:end]))
        hyde_col.add(
            ids=list(hyde_ids[start:end]),
            documents=list(hyde_texts[start:end]),
            metadatas=hyde_metas[start:end],
            embeddings=embeddings,
        )
        _logger.info(f"  HyDE: {end}/{len(valid)}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SQuAD → 向量索引构建（msmarco domain）")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "train"],
                        help="数据集分割（默认: validation）")
    parser.add_argument("--reset", action="store_true",
                        help="清空向量库后重建")
    parser.add_argument("--no-hyde", action="store_true",
                        help="跳过 HyDE 向量生成")
    parser.add_argument("--hyde-sample", type=int, default=500, metavar="N",
                        help="HyDE 段落采样上限（默认: 500）")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    ids, texts, metadatas = _load_unique_passages(args.split)

    if args.reset:
        db_path = Path(settings.chroma_db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
            _logger.info(f"已清空向量库: {db_path}")
        bm25_path = Path(settings.bm25_index_dir)
        if bm25_path.exists():
            shutil.rmtree(bm25_path)
            _logger.info(f"已清空 BM25 索引: {bm25_path}")

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
    domain = MSMarcoDomainPlugin()
    names  = domain.get_collection_names()

    col = dense_store.get_or_create_collection(names.parent)
    _index_dense(col, embedder, ids, texts, metadatas)
    _index_bm25(sparse_store, ids, texts)

    if not args.no_hyde:
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
        hyde_col = dense_store.get_or_create_collection(names.hyde)
        _index_hyde(
            hyde_col, embedder, domain, llm,
            ids, texts,
            hyde_sample=min(args.hyde_sample, len(ids)),
        )

    _logger.info("=" * 55)
    _logger.info("索引构建完成")
    _logger.info(f"  段落总数   : {len(ids)}")
    _logger.info(f"  Dense 集合 : {names.parent}")
    _logger.info(f"  BM25 路径  : {settings.bm25_index_dir}")
    _logger.info(f"  HyDE       : {'已生成' if not args.no_hyde else '跳过'}")
    _logger.info("")
    _logger.info("下一步：uv run python domains/msmarco/scripts/build_benchmark.py")


if __name__ == "__main__":
    main()
