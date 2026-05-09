from fastapi import FastAPI
from ai_app1.api.chat import router as chat_router

app = FastAPI()

app.include_router(chat_router)

@app.get("/")
def root():
    return {"msg": "AI Service Running"}