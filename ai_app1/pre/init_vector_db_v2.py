"""
Phase 1 离线索引增强：Parent-Child 架构 + 假设性问题 (HyDE)

三个 ChromaDB collection：
  android_parent  - 512字 parent chunks（喂给 LLM 的上下文）
  android_child   - 128字 child chunks（高精度向量匹配，关联 parent_id）
  android_hyde    - LLM 生成的假设性问题（关联 parent_id）

运行方式:
    uv run python -m ai_app1.pre.init_vector_db_v2
"""
import gc
import os
import re
import shutil
import time
import openai
import chromadb
from dotenv import load_dotenv

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from ai_app1.core.logger import vector_store_logger as logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CHROMA_DB_PATH = "/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db"
PARENT_CHUNK_SIZE = 512
PARENT_OVERLAP = 100
CHILD_CHUNK_SIZE = 128
CHILD_OVERLAP = 25
BATCH = 100

MINIMAX_API_KEY = os.getenv("OPENAI_API_KEY")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

MEM_WARN_PCT = 75   # 内存使用率超过此值时打印警告
MEM_GC_PCT   = 85   # 超过此值时强制 gc.collect()
MEM_ABORT_PCT = 92  # 超过此值时安全退出，避免系统 OOM


# ─── Chunking ────────────────────────────────────────────────────────────────

def _chunk_paragraphs(paragraphs, chunk_size: int, overlap: int) -> list[str]:
    """核心分块逻辑：优先保持段落边界，超长段落内部按句切分，不与其他段落混拼"""
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # 先 flush 当前积累的内容
            if current:
                chunks.append(current.strip())
                current = ""

            # 超长段落独立按句切分，每个子 chunk 独立输出，不混入前后段落
            sentences = re.split(r"(?<=[。！？!?])", para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) <= chunk_size:
                    temp += sent
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = sent
            if temp:
                chunks.append(temp.strip())
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


def chunk_file(file_path: str, chunk_size: int, overlap: int) -> list[str]:
    """流式分块：逐行读取文件，避免一次性加载大文件到内存"""
    def _iter_lines():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                para = line.strip()
                if para:
                    yield para

    return _chunk_paragraphs(_iter_lines(), chunk_size, overlap)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落分割，优先保持段落完整，超长则按句分割，带重叠"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return _chunk_paragraphs(paragraphs, chunk_size, overlap)


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



class _BufferedWriter:
    """将 (id, text, parent_id) 记录缓冲写入 ChromaDB collection，满 batch 自动 flush。"""

    def __init__(self, collection, batch: int = BATCH):
        self._col = collection
        self._batch = batch
        self._buf: list[dict] = []
        self.total = 0

    def add(self, id_: str, text: str, parent_id: str) -> None:
        self._buf.append({"id": id_, "text": text, "parent_id": parent_id})
        if len(self._buf) >= self._batch:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        self._col.add(
            documents=[r["text"] for r in self._buf],
            ids=[r["id"] for r in self._buf],
            metadatas=[{"parent_id": r["parent_id"]} for r in self._buf],
        )
        self.total += len(self._buf)
        self._buf = []

    def close(self) -> None:
        self._flush()


def _check_memory(label: str = "") -> bool:
    """检查内存用量，超阈值时 GC 或中止。返回 False 表示调用方应停止处理。"""
    if not _HAS_PSUTIL:
        return True
    pct = psutil.virtual_memory().percent
    if pct >= MEM_ABORT_PCT:
        logger.error(f"内存 {pct:.1f}% 超过中止阈值 {MEM_ABORT_PCT}%，安全退出 {label}")
        return False
    if pct >= MEM_GC_PCT:
        logger.warning(f"内存 {pct:.1f}% 超过 GC 阈值，强制回收 {label}")
        gc.collect()
    elif pct >= MEM_WARN_PCT:
        logger.info(f"内存 {pct:.1f}% 较高 {label}")
    return True

if __name__ == "__main__":
    # 0. 清理旧向量数据
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        logger.info(f"清理旧向量数据库: {CHROMA_DB_PATH}")

    # 1. 扫描 data 目录下所有文件
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not files:
        logger.warning(f"data 目录为空: {data_dir}")
        exit(0)
    logger.info(f"发现 {len(files)} 个源文件，流式索引中...")

    # 2. 初始化 ChromaDB（提前建库，边处理边写入）
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    for name in ["android_parent", "android_child", "android_hyde"]:
        try:
            db.delete_collection(name)
        except Exception:
            pass
    col_parent = db.create_collection("android_parent")
    col_child = db.create_collection("android_child")
    col_hyde = db.create_collection("android_hyde")

    # 3. 流式处理：逐文件读取 → 分块 → 直接写入，不保留全量列表
    llm = openai.OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
    parent_idx = 0
    child_writer = _BufferedWriter(col_child, BATCH)
    hyde_writer  = _BufferedWriter(col_hyde,  BATCH)


    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)
        logger.info(f"  处理: {fname} ({fsize / 1024:.1f} KB)")

        # 单个文件分块后即时处理，不累积到全局列表
        for p_text in chunk_file(fpath, PARENT_CHUNK_SIZE, PARENT_OVERLAP):
            p_id = f"p_{parent_idx}"

            # 即时写入 parent
            col_parent.add(
                documents=[p_text],
                ids=[p_id],
                metadatas=[{"chunk_idx": parent_idx, "source": fname}],
            )

            # 生成 child chunks，缓冲满 BATCH 即 flush
            for c_idx, c_text in enumerate(chunk_text(p_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)):
                child_writer.add(f"c_{parent_idx}_{c_idx}", c_text, p_id)

            # 生成 HyDE 问题，缓冲满 BATCH 即 flush
            questions = generate_hyde_questions(llm, p_text, p_id)
            for q_idx, q in enumerate(questions):
                hyde_writer.add(f"h_{parent_idx}_{q_idx}", q, p_id)

            parent_idx += 1
            if parent_idx % 10 == 0:
                logger.info(f"    已处理 {parent_idx} 个 parent chunks...")
                if not _check_memory(f"[{parent_idx} parents]"):
                    break

        # 每个文件结束后检查内存并释放
        if not _check_memory(f"[文件 {fname}]"):
            break

    # 4. flush 剩余缓冲
    child_writer.close()
    hyde_writer.close()

    logger.info(f"android_parent: 写入 {parent_idx} 条")
    logger.info(f"android_child : 写入 {child_writer.total} 条")
    logger.info(f"android_hyde  : 写入 {hyde_writer.total} 条")

    logger.info("=" * 50)
    logger.info("Phase 1 索引构建完成")
    logger.info(f"  android_parent : {parent_idx} 个")
    logger.info(f"  android_child  : {child_writer.total} 个")
    logger.info(f"  android_hyde   : {hyde_writer.total} 个")
    logger.info("下一步: uv run python -m ai_app1.pre.verify_phase1")
