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


def get_ai_client():
    return AiClient(ai_api_key=OPENAI_API_KEY)


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat(req: ChatRequest, ai_client: AiClient = Depends(get_ai_client)):
    user_id = "default_user"

    session = get_session(user_id)
    add_user_message(session, req.message)

    # 先攒着，等 AI 处理完再 summarize——避免 AI 还没看就被压缩
    messages = build_messages(session)
    reply = await ai_client.run_agent(messages)

    add_assistant_message(session, reply)

    # AI 回复后再判断是否需要 summarize
    if should_summarize(session):
        summary = await ai_client.summarize(session["history"])
        update_summary(session, summary)

    trim_history(session)

    return {"reply": reply}