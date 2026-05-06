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

user_sessions = {
    "user1": {
        "history": [],
        "summary": ""
    }
}

@router.post("/chat")
async def chat(req: ChatRequest):

    user_id = "default_user" # 先写死，后面再升级

    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": [], "summary": ""}

    history = user_sessions[user_id]["history"]

    history.append({"role":"user",
                         "content": req.message})

    summary = await summarize_history(history)

    user_sessions[user_id]["summary"] = summary

    history[:] = history[-2:]

    message = [SYSTEM_PROMPT]

    if user_sessions[user_id]["summary"]:
        message.append({
            "role": "user",
            "content": f"历史对话摘要：{user_sessions[user_id]['summary']}"
        })

    message += history

    reply = await aiClient.chat(message)

    history.append({"role":"assistant",
                         "content": reply})

    return {"reply": reply}


async def summarize_history(history):
    prompt = [
        {"role": "user", "content": "请总结以下对话的关键信息，用于后续对话参考"},
        {"role": "user", "content": str(history)}
    ]

    return await aiClient.chat(prompt)