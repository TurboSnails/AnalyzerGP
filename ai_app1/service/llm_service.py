import httpx

from ai_app1.core.config import OPENAI_API_KEY


async def chat_with_ai(messages: list):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":"application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": messages
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

    return result["choices"][0]["message"]["content"]

