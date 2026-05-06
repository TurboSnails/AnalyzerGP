from fastapi import APIRouter
from pydantic import BaseModel

from ai_app1.core.config import OPENAI_API_KEY
from ai_app1.service.AiClient import AiClient

router = APIRouter()

class ChatRequest(BaseModel):
    message: str


aiClient = AiClient(ai_api_key=OPENAI_API_KEY)

# 全局变量（先用最简单方式）
MAX_HISTORY = 6
user_sessions = {}
SYSTEM_PROMPT = {
    "role": "user",
    "content": "你是一个专业的Android开发助手，回答要简洁、准确"
}

@router.post("/chat")
async def chat(req: ChatRequest):

    user_id = "default_user" # 先写死，后面再升级

    if user_id not in user_sessions:
        user_sessions[user_id] = [] # 初始化

    history = user_sessions[user_id]

    history.append({"role":"user",
                         "content": req.message})

    history[:] = history[-MAX_HISTORY:]

    # MiniMax API 不支持 system role，把 system prompt 拼到首条 user 消息
    message = [SYSTEM_PROMPT] + history[:]

    reply = await aiClient.chat(message)

    history.append({"role":"assistant",
                         "content": reply})

    return {"reply": reply}