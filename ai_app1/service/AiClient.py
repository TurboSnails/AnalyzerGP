import json
import openai
from ai_app1.service.multiply import aiTools, TOOL_FUNCTIONS

class AiClient:
    def __init__(self, ai_api_key: str):
        self.ai_api_key = ai_api_key
        self.client = openai.OpenAI(
            base_url="https://api.minimaxi.com/v1",
            api_key=ai_api_key
        )

    async def chat(self, messages: list, use_tools: bool = False):
        try:
            kwargs = {
                "model": "MiniMax-M2.7",
                "messages": messages
            }
            if use_tools:
                kwargs["tools"] = aiTools
                print(f"[AiClient] 调用工具，messages 数量: {len(messages)}")

            response = self.client.chat.completions.create(**kwargs)
            response_message = response.choices[0].message

            if not use_tools:
                return response_message.content or ""

            if not response_message.tool_calls:
                print(f"[AiClient] 无 tool_calls，直接返回文本")
                return response_message.content or ""

            print(f"[AiClient] 检测到 tool_calls: {len(response_message.tool_calls)}")
            for tc in response_message.tool_calls:
                print(f"  - 函数: {tc.function.name}, 参数: {tc.function.arguments}")

            tool_results = []
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = TOOL_FUNCTIONS[func_name](**func_args)
                print(f"[AiClient] 执行函数 {func_name}({func_args}) = {tool_result}")
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(tool_result)
                })

            messages.append(response_message.model_dump())
            messages.extend(tool_results)

            response = self.client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=messages
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            print(f"[AiClient Error] {type(e).__name__}: {e}")
            raise

    async def summarize(self, history: list) -> str:
        prompt = [
            {"role": "user", "content": "请总结以下对话的关键信息，用于后续对话参考"},
            {"role": "user", "content": str(history)}
        ]
        return await self.chat(prompt, use_tools=False)