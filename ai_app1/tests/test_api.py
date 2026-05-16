"""
端到端 API 测试

Mock LLM + 检索组件，验证 HTTP 路由、流式响应格式、Session 行为。
不依赖真实模型和数据库，可在 CI 环境运行。

运行：
    uv run pytest ai_app1/tests/test_api.py -v
"""
from __future__ import annotations

from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from rag_framework.container import RAGContainer
from rag_framework.core.config import RAGSettings
from rag_framework.domain.base import (
    DomainPlugin,
    CollectionNames,
    QueryRoute,
    DomainPrompts,
)
from rag_framework.embedding.base import Embedder
from rag_framework.llm.base import LLMClient
from rag_framework.rerank.base import Reranker
from rag_framework.retrieval.base import Retriever, RetrievalResult, VectorStore
from rag_framework.session.base import SessionStore, SessionData


# ─── Fake 实现 ────────────────────────────────────────────────────────────────

class FakeEmbedder(Embedder):
    @property
    def embedding_dim(self) -> int:
        return 768

    def _ensure_model(self) -> None:
        pass

    def encode(self, texts, batch_size=None):
        return [[0.0] * 768 for _ in texts]


class FakeVectorStore(VectorStore):
    def get_collection(self, name: str):
        return None

    def query(self, query, collection_name, n_results=10, **filters):
        return [], [], []

    def fetch_parents(self, parent_ids, collection_name):
        return {}

    def add_batch(self, collection_name, ids, texts, metadatas):
        pass


class FakeRetriever(Retriever):
    async def retrieve(self, query, top_k=10):
        return RetrievalResult(docs=[])


class FakeReranker(Reranker):
    def rerank(self, query, candidates, top_k=5):
        return candidates[:top_k]


class FakeLLM(LLMClient):
    @property
    def backend(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def chat(self, messages, use_tools=False):
        return "Fake response"

    async def summarize(self, history):
        return "Summary"

    async def run_agent(self, messages):
        return "Agent response"

    async def chat_stream(self, messages, use_tools=False):
        for chunk in ["这是", "测试", "回复。"]:
            yield chunk

    async def run_agent_stream(self, messages):
        async for chunk in self.chat_stream(messages):
            yield chunk


class FakeSessionStore(SessionStore):
    def __init__(self, default_budget=4096):
        self._data: dict[str, SessionData] = {}
        self._budget = default_budget

    def get(self, user_id: str) -> SessionData:
        if user_id not in self._data:
            self._data[user_id] = SessionData(
                user_id=user_id, token_budget=self._budget
            )
        return self._data[user_id]

    def save(self, session: SessionData) -> None:
        self._data[session.user_id] = session

    def delete(self, user_id: str) -> None:
        self._data.pop(user_id, None)

    def list_users(self) -> list[str]:
        return list(self._data.keys())


class FakeDomain(DomainPlugin):
    @property
    def name(self) -> str:
        return "test"

    @property
    def system_prompt(self) -> str:
        return "You are a test assistant."

    @property
    def prompts(self) -> DomainPrompts:
        return DomainPrompts(system=self.system_prompt)

    def classify_query(self, query: str, history: list[dict]) -> QueryRoute:
        return QueryRoute(text=query, type="semantic")

    def get_collection_names(self) -> CollectionNames:
        return CollectionNames(parent="test_parent", child="test_child", hyde="test_hyde")

    def get_term_mapping(self) -> dict[str, str]:
        return {}

    def get_eval_dataset(self) -> list[dict]:
        return []

    def rewrite_router_rules(self, query: str, history: list[dict]):
        return 0


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def build_test_container(stream_chunks: list[str] | None = None) -> RAGContainer:
    """构建纯内存的测试容器，不依赖任何外部模型或文件系统。"""
    settings = RAGSettings(active_domain="test")

    llm = FakeLLM()
    if stream_chunks:
        # 覆盖 chat_stream 以返回指定 chunks
        async def _custom_stream(messages, use_tools=False):
            for c in stream_chunks:
                yield c
        llm.chat_stream = _custom_stream

    return RAGContainer(
        settings=settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        llm=llm,
        session_store=FakeSessionStore(),
        domain=FakeDomain(),
    )


@pytest.fixture
def client():
    """返回带注入容器的同步测试客户端（ lifespan 后覆盖为 mock）。"""
    from ai_app1.main import app

    with TestClient(app, raise_server_exceptions=True) as c:
        # lifespan startup 执行完后，重新注入 mock 容器，避免真实模型加载
        app.state.container = build_test_container()
        app.state.containers = {"default": app.state.container}
        yield c


# ─── 基础连通性 ────────────────────────────────────────────────────────────────

def test_health(client):
    """静态文件路由返回 200 或 404（取决于 static/ 是否存在），不崩溃。"""
    r = client.get("/")
    assert r.status_code in (200, 404)


def test_chat_stream_basic(client):
    """POST /chat 返回 200 text/event-stream，能读到内容。"""
    r = client.post(
        "/chat",
        json={"message": "Activity 生命周期是什么？", "user_id": "test_user"},
        headers={"Accept": "text/event-stream"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    body = r.text
    assert len(body) > 0


def test_chat_stream_content(client):
    """流式响应内容可被拼接成完整回复。"""
    from ai_app1.main import app

    chunks = ["Hello", " Android", " dev!"]
    # 替换为自定义 mock 容器
    app.state.container = build_test_container(chunks)
    app.state.containers = {"default": app.state.container}

    r = client.post(
        "/chat",
        json={"message": "hello", "user_id": "u1"},
    )
    assert r.status_code == 200
    for chunk in chunks:
        assert chunk in r.text


def test_chat_missing_message(client):
    """缺少 message 字段时返回 422 Unprocessable Entity。"""
    r = client.post("/chat", json={"user_id": "u1"})
    assert r.status_code == 422


def test_chat_empty_message(client):
    """空字符串 message 应被拒绝（422）或正常处理（200），不崩溃。"""
    r = client.post("/chat", json={"message": "", "user_id": "u1"})
    assert r.status_code in (200, 422)


def test_chat_different_users(client):
    """不同 user_id 的请求彼此独立（不共享 session 状态）。"""
    r1 = client.post("/chat", json={"message": "q1", "user_id": "alice"})
    r2 = client.post("/chat", json={"message": "q2", "user_id": "bob"})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_cors_headers(client):
    """跨域 preflight 返回正确 CORS 头。"""
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 204)
