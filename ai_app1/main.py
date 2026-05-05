from fastapi import FastAPI
from ai_app1.api.chat import router as chat_router
from ai_app1.core.config import OPENAI_API_KEY
from ai_app1.service.AiClient import AiClient

app = FastAPI()

app.include_router(chat_router)

aiClient = AiClient(ai_api_key=OPENAI_API_KEY)

@app.get("/")
def root():
    return {"msg": "AI Service Running"}