from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai_app1.core.config import OPENAI_API_KEY
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

# 模块级单例，整个进程共享一个实例
_ai_client: AiClient | None = None


def get_ai_client() -> AiClient:
    global _ai_client
    if _ai_client is None:
        _ai_client = AiClient(ai_api_key=OPENAI_API_KEY)
    return _ai_client


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat(req: ChatRequest, ai_client: AiClient = Depends(get_ai_client)):
    user_id = "default_user"

    session = get_session(user_id)
    add_user_message(session, req.message)

    messages = build_messages(session,req.message)
    reply = await ai_client.run_agent(messages)

    add_assistant_message(session, reply)

    if should_summarize(session):
        summary = await ai_client.summarize(session["history"])
        update_summary(session, summary)

    trim_history(session)

    return {"reply": reply}