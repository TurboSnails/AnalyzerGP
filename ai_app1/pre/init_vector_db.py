import chromadb

client = chromadb.PersistentClient(path="/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db")

collection = client.get_or_create_collection(name="android_docs")

collection.add(
    documents=[
        "NullPointerException：空指针 解决：检查对象是否初始化",
        "IndexOutOfBoundsException：数组越界 解决：检查下标范围"
    ],
    ids=[
        "1",
        "2"
    ],
    metadatas=[
        {
            "type": "android_crash",
            "level": "high",
            "tag": "NullPointerException"
        }
    ]
)

print("初始化完成")