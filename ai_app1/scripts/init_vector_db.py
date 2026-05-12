import os
import re
import shutil
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH
from ai_app1.retrieval.embedding import get_embedding_service


def get_project_root() -> str:
    """获取项目根目录（ai_app1 的上一级）"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chunk_text_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    按段落分割文本，返回带重叠的 chunk 列表。
    优先保证段落完整性，仅在段落超过 chunk_size 时才进一步切割。
    """
    # 先按换行分割段落
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    current_section = ""
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # 如果单个段落就超过 chunk_size，强拆
        if para_size > chunk_size:
            if current_section:
                chunks.append(current_section.strip())
                current_section = ""
                current_size = 0

            # 按句子分割（。！？.?！）
            sentences = re.split(r'(?<=[。！？!?])', para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) <= chunk_size:
                    temp += sent
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = sent
            if temp:
                current_section = temp
                current_size = len(temp)
            continue

        # 加上当前段落是否超过 size 限制
        if current_size + para_size + 1 <= chunk_size:
            current_section += para + "\n"
            current_size += para_size + 1
        else:
            if current_section:
                chunks.append(current_section.strip())
            # overlap：保留上一个 chunk 的后半部分作为开头
            if overlap > 0 and current_section:
                overlap_text = current_section[-overlap:]
                current_section = overlap_text + para + "\n"
                current_size = len(overlap_text) + para_size + 1
            else:
                current_section = para + "\n"
                current_size = para_size + 1

    if current_section.strip():
        chunks.append(current_section.strip())

    return chunks


def delete_collection():
    """删除已有的 collection 和数据库目录"""
    db_dir = CHROMA_DB_PATH
    if shutil.os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        print(f"已删除旧数据库: {db_dir}")


if __name__ == "__main__":
    delete_collection()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="android_docs")
    embed_svc = get_embedding_service()

    source_file = os.path.join(get_project_root(), "ai_app1", "Android 开发核心注意事项与避坑指南")
    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 多粒度 chunking：100、300 两种大小
    all_chunks = []
    all_ids = []
    all_metadatas = []

    for size in [100, 300]:
        chunks = chunk_text_by_size(content, chunk_size=size, overlap=int(size * 0.2))
        chunk_id_prefix = f"{size}_"
        all_chunks.extend(chunks)
        all_ids.extend([f"{chunk_id_prefix}{i}" for i in range(len(chunks))])
        all_metadatas.extend([{"chunk_size": size} for _ in chunks])
        print(f"chunk_size={size}: {len(chunks)} 个 chunk")

    embeddings = embed_svc.encode(all_chunks, batch_size=32)
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings,
    )

    print(f"初始化完成，共 {len(all_chunks)} 个 chunk")