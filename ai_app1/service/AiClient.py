import anthropic


class AiClient:
    def __init__(self, ai_api_key: str):
        self.ai_api_key = ai_api_key
        # 关键点：必须确保这里使用的是 AsyncAnthropic
        self.aiClient = anthropic.AsyncAnthropic(
            base_url="https://api.minimaxi.com/anthropic",
            api_key=ai_api_key
        )

    async def chat(self, messages: list):
        try:
            message = await self.aiClient.messages.create(
                model="MiniMax-M2.7",
                max_tokens=1000,
                messages=messages
            )

            for block in message.content:
                if block.type == "text":
                    return block.text
            return "No text response."
        except Exception as e:
            print(f"[AiClient Error] {type(e).__name__}: {e}")
            raise