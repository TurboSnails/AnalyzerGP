"""
Phase 1 离线索引增强：Parent-Child 架构 + 假设性问题 (HyDE)

三个 ChromaDB collection：
  android_parent  - 512字 parent chunks（喂给 LLM 的上下文）
  android_child   - 128字 child chunks（高精度向量匹配，关联 parent_id）
  android_hyde    - LLM 生成的假设性问题（关联 parent_id）

运行方式:
    uv run python -m ai_app1.pre.init_vector_db_v2
"""
import os
import re
import time
import logging
import openai
import chromadb
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_v2")

CHROMA_DB_PATH = "/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db"
PARENT_CHUNK_SIZE = 512
PARENT_OVERLAP = 100
CHILD_CHUNK_SIZE = 128
CHILD_OVERLAP = 25

MINIMAX_API_KEY = os.getenv("OPENAI_API_KEY")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL = "MiniMax-M2.7"


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落分割，优先保持段落完整，超长则按句分割，带重叠"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            sentences = re.split(r"(?<=[。！？!?])", para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) <= chunk_size:
                    temp += sent
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = sent
            current = temp
        elif len(current) + len(para) + 1 <= chunk_size:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if overlap > 0 and current else ""
            current = overlap_text + para + "\n"

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


# ─── HyDE 问题生成 ────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """去除 MiniMax 思维链 <think>...</think> 标签及其内容"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_questions(text: str) -> list[str]:
    """从模型输出中提取有效问题行（过滤思维链残留和空行）"""
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        # 去除行首的 - / • / 引号 / 序号
        line = re.sub(r'^[-•\d.、"「\s]+|["」\s]+$', "", line).strip()
        # 保留：长度 > 5 且以句号/问号结尾，或包含问号
        if len(line) > 5 and ("？" in line or "?" in line or line.endswith(("。", "吗", "呢"))):
            questions.append(line)
    return questions[:3]


def generate_hyde_questions(client: openai.OpenAI, chunk: str, parent_id: str, retry: int = 3) -> list[str]:
    """为 parent chunk 生成 3 个开发者可能提问的假设性问题"""
    prompt = (
        f"你是Android开发专家。以下是一段Android开发文档：\n\n{chunk}\n\n"
        "请生成3个开发者可能会问的问题，这些问题可以通过上述文档内容回答。\n"
        "要求：直接输出3个问题，每行一个，不要编号，不要额外说明，不要思考过程。"
    )
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=MINIMAX_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            cleaned = _strip_thinking(raw)

            # 优先从清洗后内容提取
            questions = _extract_questions(cleaned)

            # 降级1：清洗后内容取前3行
            if not questions and cleaned:
                questions = [l.strip() for l in cleaned.split("\n") if l.strip()][:3]

            # 降级2：若 <think> 未闭合导致 cleaned 为空，从 raw 中提取含问号的行
            if not questions:
                q_lines = [l.strip() for l in raw.split("\n")
                           if ("？" in l or "?" in l) and len(l.strip()) > 5]
                questions = q_lines[:3]

            logger.debug(f"  HyDE [{parent_id}]: {questions}")
            return questions
        except Exception as e:
            logger.warning(f"  HyDE [{parent_id}] 第{attempt + 1}次失败: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
    return []


# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 读取源文件
    src = os.path.join(os.path.dirname(__file__), "..", "Android 开发核心注意事项与避坑指南")
    with open(src, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"源文件加载: {len(content)} 字符")

    # 2. 构建 parent chunks
    parent_chunks = chunk_text(content, PARENT_CHUNK_SIZE, PARENT_OVERLAP)
    logger.info(f"Parent chunks: {len(parent_chunks)} 个 (size={PARENT_CHUNK_SIZE}, overlap={PARENT_OVERLAP})")

    # 3. 构建 child chunks（每个 parent 内部细分）
    child_records: list[dict] = []
    for p_idx, p_text in enumerate(parent_chunks):
        children = chunk_text(p_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
        for c_idx, c_text in enumerate(children):
            child_records.append({
                "id": f"c_{p_idx}_{c_idx}",
                "text": c_text,
                "parent_id": f"p_{p_idx}",
            })
    logger.info(f"Child chunks: {len(child_records)} 个 (size={CHILD_CHUNK_SIZE}, overlap={CHILD_OVERLAP})")

    # 4. 生成 HyDE 假设性问题（每 parent 3 个）
    llm = openai.OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
    hyde_records: list[dict] = []
    logger.info(f"开始生成 HyDE 问题 ({len(parent_chunks)} 个 parent)...")
    for p_idx, p_text in enumerate(parent_chunks):
        p_id = f"p_{p_idx}"
        questions = generate_hyde_questions(llm, p_text, p_id)
        for q_idx, q in enumerate(questions):
            hyde_records.append({
                "id": f"h_{p_idx}_{q_idx}",
                "text": q,
                "parent_id": p_id,
            })
        logger.info(f"  [{p_idx + 1}/{len(parent_chunks)}] {p_id}: {len(questions)} 个问题")
        time.sleep(0.3)  # 避免触发限速
    logger.info(f"HyDE 问题总计: {len(hyde_records)} 个")

    # 5. 写入 ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    for name in ["android_parent", "android_child", "android_hyde"]:
        try:
            db.delete_collection(name)
            logger.info(f"删除旧 collection: {name}")
        except Exception:
            pass

    col_parent = db.create_collection("android_parent")
    col_child = db.create_collection("android_child")
    col_hyde = db.create_collection("android_hyde")

    # 写入 parent
    col_parent.add(
        documents=parent_chunks,
        ids=[f"p_{i}" for i in range(len(parent_chunks))],
        metadatas=[{"chunk_idx": i} for i in range(len(parent_chunks))],
    )
    logger.info(f"android_parent: 写入 {len(parent_chunks)} 条")

    # 批量写入 child（每批 100 条）
    BATCH = 100
    for i in range(0, len(child_records), BATCH):
        batch = child_records[i : i + BATCH]
        col_child.add(
            documents=[r["text"] for r in batch],
            ids=[r["id"] for r in batch],
            metadatas=[{"parent_id": r["parent_id"]} for r in batch],
        )
    logger.info(f"android_child: 写入 {len(child_records)} 条")

    # 写入 hyde
    if hyde_records:
        col_hyde.add(
            documents=[r["text"] for r in hyde_records],
            ids=[r["id"] for r in hyde_records],
            metadatas=[{"parent_id": r["parent_id"]} for r in hyde_records],
        )
    logger.info(f"android_hyde: 写入 {len(hyde_records)} 条")

    logger.info("=" * 50)
    logger.info("Phase 1 索引构建完成")
    logger.info(f"  android_parent : {len(parent_chunks)} 个")
    logger.info(f"  android_child  : {len(child_records)} 个")
    logger.info(f"  android_hyde   : {len(hyde_records)} 个")
    logger.info("下一步: uv run python -m ai_app1.pre.verify_phase1")
