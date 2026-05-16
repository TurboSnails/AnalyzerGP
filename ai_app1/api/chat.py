"""
Chat API 路由 — 基于 rag_framework 的流式对话接口（多领域支持）

职责：
  1. 接收用户消息与可选领域参数
  2. 支持单领域路由（自动/显式）与多领域融合（domain="all"）
  3. 返回 StreamingResponse

容器从 app.state.containers / app.state.container 获取，零全局状态。
"""
import asyncio
import time
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_framework.container import RAGContainer
from rag_framework.core.logger import chat_logger
from rag_framework.retrieval.base import RetrievedDoc

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    domain: str = ""  # "msmarco", "android", "all", 空字符串自动路由


def _has_chinese(text: str) -> bool:
    """检查字符串是否包含中文字符。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _resolve_single_container(
    containers: dict[str, RAGContainer],
    default_container: RAGContainer,
    domain: str,
    query: str = "",
) -> RAGContainer:
    """单领域容器解析。"""
    # 1. 用户显式指定
    if domain and domain in containers:
        return containers[domain]

    # 2. 自动路由：含中文 → android
    if query and _has_chinese(query):
        android = containers.get("android")
        if android is not None:
            return android

    # 3. 回退默认
    return default_container


async def _retrieve_from_domain(
    container: RAGContainer, query: str, user_id: str
) -> list[RetrievedDoc]:
    """从单个领域异步检索，返回文档列表。"""
    session = container.session_store.get(user_id)
    routes = container.build_routes(query, session.history)
    result = await container.retriever.retrieve(routes)
    return result.docs


async def _multi_domain_chat_stream(
    containers: dict[str, RAGContainer],
    default_container: RAGContainer,
    query: str,
    user_id: str,
):
    """
    多领域融合对话流。

    1. 并行从所有领域检索
    2. 合并去重、按分数排序、截取 top_k
    3. 使用默认领域的 system_prompt + session + LLM 生成回答
    4. 会话历史保存到默认领域
    """
    # 1. 并行检索所有领域
    tasks = [
        _retrieve_from_domain(c, query, user_id)
        for c in containers.values()
    ]
    all_docs_per_domain = await asyncio.gather(*tasks)

    # 2. 合并去重 + 排序
    seen: set[str] = set()
    merged: list[RetrievedDoc] = []
    for docs in all_docs_per_domain:
        for d in docs:
            if d.id not in seen:
                seen.add(d.id)
                merged.append(d)

    if not merged:
        yield "未检索到任何相关文档，请尝试更换关键词。"
        return

    merged.sort(key=lambda x: x.score, reverse=True)
    top_docs = merged[:6]  # 限制上下文长度，避免超出 token 预算

    # 3. 使用默认领域构建 messages
    session = default_container.session_store.get(user_id)
    messages: list[dict] = [
        {"role": "system", "content": default_container.domain.system_prompt}
    ]
    if session.summary:
        messages.append(
            {"role": "user", "content": f"【历史摘要】{session.summary}"}
        )
    messages.extend(session.history)

    context = "\n\n".join(
        f"[{idx + 1}] {d.text}" for idx, d in enumerate(top_docs)
    )
    messages.append(
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{query}",
        }
    )

    # 4. 流式调用 LLM（共享组件）
    full_reply = ""
    async for chunk in default_container.llm.chat_stream(messages, use_tools=False):
        full_reply += chunk
        yield chunk

    # 5. 维护默认领域会话历史
    session.history.append({"role": "user", "content": query})
    session.history.append({"role": "assistant", "content": full_reply})
    default_container.session_store.save(session)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    流式聊天接口。

    请求：
      { "message": "...", "user_id": "...", "domain": "msmarco" }
      domain 可选值：
        - ""       : 自动路由（中文→android，其他→默认）
        - "all"    : 同时检索所有领域并融合回答
        - 具体名称 : 强制使用指定领域

    响应：text/event-stream 流式输出
    """
    req_id = id(req)
    containers: dict[str, RAGContainer] = getattr(
        request.app.state, "containers", {}
    ) or {}
    default_container: RAGContainer = getattr(
        request.app.state, "container", None
    )

    # 兼容旧测试：仅设置 container 但未设置 containers 时自动包装
    if not containers and default_container is not None:
        containers = {"default": default_container}

    if not containers or default_container is None:
        raise RuntimeError("RAGContainer 未初始化，请检查 lifespan 是否已执行")

    # 确定目标领域
    is_multi_domain = req.domain.strip().lower() == "all"
    if is_multi_domain:
        resolved_domain = "all"
    else:
        container = _resolve_single_container(
            containers, default_container, req.domain, req.message
        )
        resolved_domain = container.domain.name if container.domain else "default"

    chat_logger.info(
        f"[{req_id}] 收到请求: message={req.message[:50]!r}, "
        f"user={req.user_id}, domain={req.domain or 'auto'} "
        f"→ 实际使用: {resolved_domain}"
    )
    start = time.monotonic()

    async def content_generator() -> AsyncIterator[str]:
        t_before = time.monotonic()
        first_chunk_logged = False

        if is_multi_domain:
            stream = _multi_domain_chat_stream(
                containers, default_container, req.message, req.user_id
            )
        else:
            stream = container.chat_stream(req.message, req.user_id)

        async for chunk in stream:
            if chunk:
                if not first_chunk_logged:
                    chat_logger.info(
                        f"[{req_id}] 收到 LLM 首字: "
                        f"TTFT={1000*(time.monotonic()-t_before):.0f}ms"
                    )
                    first_chunk_logged = True
                yield chunk

        elapsed = time.monotonic() - start
        chat_logger.info(f"[{req_id}] 流式传输完成: 总耗时={elapsed:.2f}s")

    return StreamingResponse(content_generator(), media_type="text/event-stream")
