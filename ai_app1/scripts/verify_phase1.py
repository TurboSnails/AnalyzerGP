"""
Phase 1 验收测试：验证 Parent-Child + HyDE 索引是否正确构建

检查项：
  1. 三个 collection 均存在
  2. 数量合理：child >= parent, hyde >= parent
  3. 字段完整：child 有 parent_id, hyde 有 parent_id
  4. 向量检索可用：child/hyde 均能返回结果
  5. parent_id 引用有效：hyde 里的 parent_id 能在 parent 中找到

运行方式:
    uv run python -m ai_app1.pre.verify_phase1
"""
import sys
import logging
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_p1")
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    logger.info(f"{status} {name}{(' — ' + detail) if detail else ''}")
    return condition


def run() -> bool:
    results: list[bool] = []
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    existing = {c.name for c in db.list_collections()}
    logger.info(f"已有 collections: {sorted(existing)}")

    # ── 1. collection 存在 ────────────────────────────────────────────────────
    for name in ["android_parent", "android_child", "android_hyde"]:
        ok = check(f"collection '{name}' 存在", name in existing)
        results.append(ok)

    if not all(results):
        logger.error("基本 collection 缺失，请先运行 init_vector_db_v2.py")
        return False

    parent = db.get_collection("android_parent")
    child = db.get_collection("android_child")
    hyde = db.get_collection("android_hyde")
    p_cnt = parent.count()
    c_cnt = child.count()
    h_cnt = hyde.count()
    logger.info(f"数量: parent={p_cnt}, child={c_cnt}, hyde={h_cnt}")

    # ── 2. 数量合理性 ─────────────────────────────────────────────────────────
    results.append(check("parent 数量 >= 5", p_cnt >= 5, str(p_cnt)))
    results.append(check("child >= parent (细粒度切割)", c_cnt >= p_cnt, f"{c_cnt} >= {p_cnt}"))
    results.append(check("hyde >= parent (至少1问题/parent)", h_cnt >= p_cnt, f"{h_cnt} >= {p_cnt}"))

    # ── 3. 字段完整性 ─────────────────────────────────────────────────────────
    c_sample = child.get(limit=5)
    ok_child_meta = all("parent_id" in m for m in c_sample["metadatas"])
    results.append(check("child metadata 含 parent_id", ok_child_meta))

    h_sample = hyde.get(limit=5)
    ok_hyde_meta = all("parent_id" in m for m in h_sample["metadatas"])
    results.append(check("hyde metadata 含 parent_id", ok_hyde_meta))

    # ── 4. 向量检索可用 ───────────────────────────────────────────────────────
    test_queries = ["Activity内存泄漏", "ANR问题", "Fragment事务"]
    for q in test_queries:
        r_child = child.query(query_texts=[q], n_results=3)
        ok_c = len(r_child["documents"][0]) > 0
        results.append(check(f"child 检索 '{q}'", ok_c, f"{len(r_child['documents'][0])} 结果"))

        r_hyde = hyde.query(query_texts=[q], n_results=3)
        ok_h = len(r_hyde["documents"][0]) > 0
        results.append(check(f"hyde 检索 '{q}'", ok_h, f"{len(r_hyde['documents'][0])} 结果"))

    # ── 5. parent_id 引用有效 ─────────────────────────────────────────────────
    parent_ids_set = set(parent.get()["ids"])
    hyde_pids = {m["parent_id"] for m in hyde.get()["metadatas"]}
    dangling = hyde_pids - parent_ids_set
    results.append(check("hyde parent_id 引用无悬空", len(dangling) == 0,
                         f"{len(dangling)} 个悬空引用" if dangling else ""))

    # ── 抽样展示 ──────────────────────────────────────────────────────────────
    logger.info("\n── 抽样: Parent chunks ──")
    p_sample = parent.get(limit=2)
    for pid, ptext in zip(p_sample["ids"], p_sample["documents"]):
        logger.info(f"  [{pid}] {ptext[:120]!r}...")

    logger.info("\n── 抽样: HyDE 假设性问题 ──")
    for hid, htext, hmeta in zip(h_sample["ids"], h_sample["documents"], h_sample["metadatas"]):
        logger.info(f"  [{hid}] → {hmeta['parent_id']}: {htext!r}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    logger.info(f"\n{'=' * 55}")
    logger.info(f"Phase 1 验收: {passed}/{total} 通过 {'✅ 全部通过' if passed == total else '⚠️ 存在失败项'}")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
