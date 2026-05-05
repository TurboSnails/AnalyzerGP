from fastapi import APIRouter
from pydantic import BaseModel

from ai_app1.core.config import OPENAI_API_KEY
from ai_app1.service.AiClient import AiClient
from ai_app1.service.llm_service import chat_with_ai

router = APIRouter()

class ChatRequest(BaseModel):
    message: str


aiClient = AiClient(ai_api_key=OPENAI_API_KEY)

# 全局变量（先用最简单方式）
chat_history = []

@router.post("/chat")
async def chat(req: ChatRequest):

    chat_history.append({"role":"user",
                         "content": req.message})

    reply = await aiClient.chat(chat_history)

    chat_history.append({"role":"assistant",
                         "content": reply})

    return {"reply": reply}