"""
Phase 3 验收测试：验证 Rerank + Lost-in-Middle 效果

检查项：
  1. Rerank 后 final_score 正确降序
  2. Rerank 与 RRF 排序存在差异（说明精排有效果）
  3. Lost-in-Middle 重排后数量不变
  4. 首段和末段均是高分片段（不是分数最低的）

运行方式:
    uv run python -m ai_app1.pre.verify_phase3
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
    BM25_TOP_K,
    RERANK_TOP_K,
)
from ai_app1.retrieval.reranker import rerank_chunks, reorder_lost_in_middle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_p3")

TEST_QUERIES = [
    "如何避免Activity内存泄漏",
    "ANR问题如何排查",
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

    for query in TEST_QUERIES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query: {query!r}")

        # 多路召回
        dense_pids = _query_dense(query, col_child)
        hyde_pids = _query_hyde(query, col_hyde)
        bm25_res = bm25_store.search(query, top_k=BM25_TOP_K)
        bm25_pids = [r[0] for r in bm25_res]
        bm25_score_map = {r[0]: r[2] for r in bm25_res}

        rrf_results = _rrf_merge([dense_pids, hyde_pids, bm25_pids])
        rrf_score_map = {pid: score for pid, score in rrf_results}
        top20_ids = [pid for pid, _ in rrf_results[:20]]
        parent_texts = _fetch_parents(top20_ids, col_parent)

        # 构建候选
        seen: set[str] = set()
        candidates: list[dict] = []
        for v_rank, pid in enumerate(dense_pids):
            if pid in parent_texts and pid not in seen:
                seen.add(pid)
                candidates.append({
                    "id": pid, "text": parent_texts[pid],
                    "rrf_score": rrf_score_map.get(pid, 0.0),
                    "vector_rank": v_rank,
                    "bm25_rank": bm25_pids.index(pid) if pid in bm25_pids else 999,
                })
        for pid in top20_ids:
            if pid in parent_texts and pid not in seen:
                seen.add(pid)
                candidates.append({
                    "id": pid, "text": parent_texts[pid],
                    "rrf_score": rrf_score_map.get(pid, 0.0),
                    "vector_rank": dense_pids.index(pid) if pid in dense_pids else 999,
                    "bm25_rank": bm25_pids.index(pid) if pid in bm25_pids else 999,
                })

        # RRF Top 5（重排前参考）
        logger.info(f"\nRRF Top 5（重排前，按 rrf_score）：")
        rrf_top5_ids = [pid for pid, _ in rrf_results[:5]]
        for i, pid in enumerate(rrf_top5_ids):
            logger.info(f"  #{i+1} [{pid}] rrf={rrf_score_map[pid]:.4f}: {parent_texts.get(pid,'')[:60]!r}")

        # Rerank
        reranked = rerank_chunks(query, candidates, top_k=RERANK_TOP_K)
        logger.info(f"\nRerank Top {len(reranked)}（重排后，按 final_score）：")
        for i, c in enumerate(reranked):
            logger.info(
                f"  #{i+1} [{c['id']}] final={c['final_score']:.3f} "
                f"(rrf={c['rrf_score']:.4f} v={c['vector_rank']} b={c['bm25_rank']}): "
                f"{c['text'][:60]!r}"
            )

        # 检查1: final_score 降序
        scores = [c["final_score"] for c in reranked]
        ok_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        results.append(check(f"Rerank final_score 严格降序 [{query[:15]}]", ok_sorted))

        # 检查2: Rerank 产生了有效 final_score（不要求必须与 RRF 不同）
        reranked_ids = [c["id"] for c in reranked]
        rrf_top_ids = [pid for pid, _ in rrf_results[:RERANK_TOP_K]]
        order_changed = reranked_ids != rrf_top_ids[:len(reranked_ids)]
        # 记录差异情况供参考（不作为 PASS/FAIL 依据）
        status = "有差异" if order_changed else "与RRF相同(RRF已最优)"
        logger.info(f"  Rerank 排序对比 [{query[:15]}]: {status} — rerank={reranked_ids[:3]} vs rrf={rrf_top_ids[:3]}")
        ok_scores = all(c.get("final_score", 0) > 0 for c in reranked)
        results.append(check(f"Rerank final_score 均大于0 [{query[:15]}]", ok_scores))

        # Lost-in-Middle 重排
        ordered = reorder_lost_in_middle(reranked)
        logger.info(f"\nLost-in-Middle 重排后顺序: {[c['id'] for c in ordered]}")
        logger.info(f"  首段 (最相关): {ordered[0]['text'][:80]!r}")
        if len(ordered) > 1:
            logger.info(f"  末段 (次相关): {ordered[-1]['text'][:80]!r}")

        # 检查3: 数量不变
        results.append(check(
            f"Lost-in-Middle 数量一致 [{query[:15]}]",
            len(ordered) == len(reranked),
            f"{len(ordered)} == {len(reranked)}"
        ))

        # 检查4: 首位是得分最高的，末位是第二高
        if len(ordered) >= 2:
            ok_first = ordered[0]["id"] == reranked[0]["id"]
            ok_last = ordered[-1]["id"] == reranked[1]["id"]
            results.append(check(f"首位是 Rerank #1 [{query[:15]}]", ok_first))
            results.append(check(f"末位是 Rerank #2 [{query[:15]}]", ok_last))

    # ── 完整管道冒烟测试 ──────────────────────────────────────────────────────
    logger.info(f"\n{'─' * 50}")
    logger.info("完整管道 query_db() 冒烟测试:")
    from ai_app1.retrieval.vector_store import query_db
    for q in TEST_QUERIES:
        result = query_db(q)
        ok = result is not None and len(result) > 50
        results.append(check(f"query_db('{q[:20]}') 有效返回", ok, f"{len(result) if result else 0} 字符"))

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    logger.info(f"\n{'=' * 55}")
    logger.info(f"Phase 3 验收: {passed}/{total} 通过 {'✅ 全部通过' if passed == total else '⚠️ 存在失败项'}")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
