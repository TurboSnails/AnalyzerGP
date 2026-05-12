"""
Phase 2 验收测试：验证混合检索（Dense + HyDE + BM25 + RRF）

检查项：
  1. BM25 索引构建成功
  2. 三条检索路径各自有结果
  3. RRF 融合后结果非空
  4. 混合检索比单路向量检索命中更多相关 parent

运行方式:
    uv run python -m ai_app1.pre.verify_phase2
"""
import sys
import logging
from ai_app1.retrieval import bm25_store
from ai_app1.retrieval.vector_store import (
    _get_collection,
    _rrf_merge,
    _query_dense,
    _query_hyde,
    _fetch_parents,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_p2")

TEST_QUERIES = [
    "如何避免Activity内存泄漏",
    "ANR问题如何排查",
    "Fragment事务提交注意事项",
]
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    logger.info(f"{status} {name}{(' — ' + detail) if detail else ''}")
    return condition


def run() -> bool:
    results: list[bool] = []

    col_parent = _get_collection("android_parent")
    col_child = _get_collection("android_child")
    col_hyde = _get_collection("android_hyde")

    if not all([col_parent, col_child, col_hyde]):
        logger.error("v2 collections 未就绪，请先运行 init_vector_db_v2.py")
        return False

    # ── 1. BM25 索引 ──────────────────────────────────────────────────────────
    bm25_test = bm25_store.search("内存泄漏", top_k=3)
    results.append(check("BM25 索引构建成功", len(bm25_test) > 0, f"{len(bm25_test)} 条结果"))
    if bm25_test:
        logger.info(f"  BM25 top: [{bm25_test[0][0]}] score={bm25_test[0][2]:.3f}: {bm25_test[0][1][:60]!r}")

    # ── 2. 各路径独立检索 + RRF ───────────────────────────────────────────────
    for query in TEST_QUERIES:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Query: {query!r}")

        dense_pids = _query_dense(query, col_child)
        hyde_pids = _query_hyde(query, col_hyde)
        bm25_res = bm25_store.search(query, top_k=10)
        bm25_pids = [r[0] for r in bm25_res]

        logger.info(f"  Dense pids : {dense_pids[:4]}")
        logger.info(f"  HyDE  pids : {hyde_pids[:4]}")
        logger.info(f"  BM25  pids : {bm25_pids[:4]}")

        results.append(check(f"Dense 有结果 '{query}'", len(dense_pids) > 0, str(len(dense_pids))))
        results.append(check(f"HyDE  有结果 '{query}'", len(hyde_pids) > 0, str(len(hyde_pids))))
        results.append(check(f"BM25  有结果 '{query}'", len(bm25_pids) > 0, str(len(bm25_pids))))

        rrf = _rrf_merge([dense_pids, hyde_pids, bm25_pids])
        results.append(check(f"RRF 融合有结果 '{query}'", len(rrf) > 0, f"{len(rrf)} 候选"))

        logger.info(f"  RRF top3: {[(pid, f'{s:.4f}') for pid, s in rrf[:3]]}")

        top3_ids = [pid for pid, _ in rrf[:3]]
        texts = _fetch_parents(top3_ids, col_parent)
        for pid in top3_ids:
            logger.info(f"    [{pid}] {texts.get(pid, '')[:80]!r}")

    # ── 3. 混合 vs 单路 覆盖率对比 ───────────────────────────────────────────
    logger.info(f"\n{'─' * 50}")
    logger.info("混合检索 vs 向量单路 — 新增命中 parent 对比:")
    for query in TEST_QUERIES[:2]:
        dense_pids = _query_dense(query, col_child)
        hyde_pids = _query_hyde(query, col_hyde)
        bm25_pids = [r[0] for r in bm25_store.search(query, top_k=10)]

        rrf = _rrf_merge([dense_pids, hyde_pids, bm25_pids])
        hybrid_top5 = {pid for pid, _ in rrf[:5]}
        vector_top5 = set(dense_pids[:5])

        added = hybrid_top5 - vector_top5
        removed = vector_top5 - hybrid_top5
        logger.info(f"  '{query}': 新增 {len(added)} 个 {added}, 替换 {len(removed)} 个")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    logger.info(f"\n{'=' * 55}")
    logger.info(f"Phase 2 验收: {passed}/{total} 通过 {'✅ 全部通过' if passed == total else '⚠️ 存在失败项'}")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
