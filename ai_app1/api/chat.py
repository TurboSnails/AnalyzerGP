"""
Chat API 路由：处理 /chat POST 请求。

请求流程：
    1. 接收用户消息，加入会话 history
    2. 构建 messages（系统提示 + 摘要 + 历史 + 文档检索结果）
    3. 调用 run_agent 获取 AI 回复
    4. AI 回复后判断是否需要 summarize（token 预算耗尽触发）
    5. 裁剪 history 到 MAX_HISTORY 条
    6. 返回 AI 回复

AiClient 使用进程级单例，避免重复创建 OpenAI 客户端实例。
"""
import time
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_app1.core.config import OPENAI_API_KEY
from ai_app1.core.logger import chat_logger
from ai_app1.service.AiClient import AiClient
from ai_app1.service.session import (
    get_session,
    add_user_message,
    add_assistant_message,
    update_summary,
    trim_history,
    build_messages,
    should_summarize,
)

router = APIRouter()

# ─── AiClient 单例 ───────────────────────────────────────────
# 懒汉式单例：进程内全局共享，复用 HTTP 连接池
_ai_client: AiClient | None = None


def get_ai_client() -> AiClient:
    """
    获取或创建 AiClient 单例。

    首次调用时创建实例，后续调用直接返回，节省每次请求创建客户端的开销。
    """
    global _ai_client
    if _ai_client is None:
        _ai_client = AiClient(ai_api_key=OPENAI_API_KEY)
        chat_logger.info(f"AiClient 单例初始化完成: id={id(_ai_client)}")
    return _ai_client


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    message: str


@router.post("/chat")
async def chat(req: ChatRequest, ai_client: AiClient = Depends(get_ai_client)):
    """
    处理用户聊天请求的主入口。流式响应，用户首个 token 即可见。

    Args:
        req: 包含 message 字段的请求体
        ai_client: 通过 Depends 注入的 AiClient 单例

    Returns:
        StreamingResponse 流式输出 AI 回复
    """
    req_id = id(req)
    chat_logger.info(f"[{req_id}] 收到请求: message={req.message[:50]!r}")
    start = time.monotonic()

    user_id = "default_user"

    # 获取或创建会话
    session = get_session(user_id)
    chat_logger.debug(f"[{req_id}] 获取 session: history_len={len(session['history'])}, summary={'有' if session['summary'] else '无'}")

    # 1. 用户消息入栈
    add_user_message(session, req.message)
    chat_logger.debug(f"[{req_id}] 用户消息已添加: history_len={len(session['history'])}")

    # 2. 构建发送给 LLM 的完整消息列表
    messages = build_messages(session, req.message)
    chat_logger.debug(f"[{req_id}] messages 构建完成: {len(messages)} 条")

    # 3. 流式收集完整回复（用于 summarize 判断），同时流式 yield 给客户端
    full_reply_parts: list[str] = []

    async def content_generator():
        nonlocal messages
        async for chunk in ai_client.stream_run_agent(messages):
            if chunk:
                full_reply_parts.append(chunk)
                yield chunk

        # 流式结束后更新 session（保证 summarize 在 AI 回复之后触发）
        reply = "".join(full_reply_parts)
        add_assistant_message(session, reply)
        chat_logger.debug(f"[{req_id}] 助手消息已添加: history_len={len(session['history'])}")

        if should_summarize(session):
            chat_logger.info(f"[{req_id}] 开始 summarize")
            summary = await ai_client.summarize(session["history"])
            update_summary(session, summary)
            chat_logger.info(f"[{req_id}] summarize 完成: summary_len={len(summary)}")
        else:
            chat_logger.debug(f"[{req_id}] 跳过 summarize: should_summarize=False")

        trim_history(session)
        chat_logger.debug(f"[{req_id}] trim_history 完成: history_len={len(session['history'])}, trimmed_len={len(session['trimmed'])}")

        elapsed = time.monotonic() - start
        chat_logger.info(f"[{req_id}] 请求处理完成: 总耗时={elapsed:.2f}")

    return StreamingResponse(content_generator(), media_type="text/plain")