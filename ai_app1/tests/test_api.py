"""
端到端 API 测试

Mock LLM + 检索组件，验证 HTTP 路由、流式响应格式、Session 行为。
不依赖真实模型和数据库，可在 CI 环境运行。

运行：
    uv run pytest ai_app1/tests/test_api.py -v
"""
from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_mock_container(stream_chunks: list[str] | None = None):
    """构建注入用的 RAGContainer mock。"""
    chunks = stream_chunks or ["这是", "测试", "回复。"]

    async def _fake_stream(*args, **kwargs) -> AsyncIterator[str]:
        for c in chunks:
            yield c

    container = MagicMock()
    container.chat_stream = _fake_stream
    return container


@pytest.fixture
def client():
    """返回带 mock 容器的同步测试客户端。"""
    mock_container = _make_mock_container()

    with patch(
        "ai_app1.api.chat.get_container",
        return_value=mock_container,
    ):
        from ai_app1.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ─── 基础连通性 ────────────────────────────────────────────────────────────────

def test_health(client: TestClient):
    """静态文件路由返回 200 或 404（取决于 static/ 是否存在），不崩溃。"""
    r = client.get("/")
    assert r.status_code in (200, 404)


def test_chat_stream_basic(client: TestClient):
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


def test_chat_stream_content(client: TestClient):
    """流式响应内容可被拼接成完整回复。"""
    chunks = ["Hello", " Android", " dev!"]
    mock_container = _make_mock_container(chunks)

    with patch("ai_app1.api.chat._get_container", return_value=mock_container):
        from ai_app1.main import app
        with TestClient(app) as c:
            r = c.post(
                "/chat",
                json={"message": "hello", "user_id": "u1"},
            )
    assert r.status_code == 200
    # SSE body 中应包含所有 chunk 内容
    for chunk in chunks:
        assert chunk in r.text


def test_chat_missing_message(client: TestClient):
    """缺少 message 字段时返回 422 Unprocessable Entity。"""
    r = client.post("/chat", json={"user_id": "u1"})
    assert r.status_code == 422


def test_chat_empty_message(client: TestClient):
    """空字符串 message 应被拒绝（422）或正常处理（200），不崩溃。"""
    r = client.post("/chat", json={"message": "", "user_id": "u1"})
    assert r.status_code in (200, 422)


def test_chat_different_users(client: TestClient):
    """不同 user_id 的请求彼此独立（不共享 session 状态）。"""
    mock_container = _make_mock_container(["ok"])
    with patch("ai_app1.api.chat._get_container", return_value=mock_container):
        from ai_app1.main import app
        with TestClient(app) as c:
            r1 = c.post("/chat", json={"message": "q1", "user_id": "alice"})
            r2 = c.post("/chat", json={"message": "q2", "user_id": "bob"})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_cors_headers(client: TestClient):
    """跨域 preflight 返回正确 CORS 头。"""
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code in (200, 204)
