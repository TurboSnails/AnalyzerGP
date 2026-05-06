from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai_app1.core.config import OPENAI_API_KEY
from ai_app1.service.AiClient import AiClient
from ai_app1.service.session import get_session, add_user_message, add_assistant_message, update_summary, trim_history, build_messages

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

    summary = await ai_client.summarize(session["history"])
    update_summary(session, summary)
    trim_history(session)

    messages = build_messages(session)
    reply = await ai_client.chat(messages, use_tools=True)

    add_assistant_message(session, reply)

    return {"reply": reply}