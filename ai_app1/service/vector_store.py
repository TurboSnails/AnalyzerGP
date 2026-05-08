import chromadb
from ai_app1.core.config import CHROMA_DB_PATH

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="android_docs")

MAX_DISTANCE = 1.2


def query_db(query: str):
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    valid_docs = []
    for doc, distance in zip(docs, distances):
        print(f"--》distance: {distance}, doc: {doc[:50]}...")
        if distance <= MAX_DISTANCE:
            valid_docs.append(doc)

    if not valid_docs:
        return None

    return "\n".join(valid_docs)