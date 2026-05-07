"""
AiClient：封装与 MiniMax-M2.7 模型的交互逻辑。

提供两个核心接口：
- chat(messages, use_tools=False)：普通对话，返回文本
- run_agent(messages)：工具增强对话，内部自动处理工具调用循环

内部使用 openai.OpenAI SDK，通过 base_url 指向 MiniMax 兼容端点。
每次请求不会重新创建客户端（AiClient 单例由 chat.py 管理）。
"""
import json
import logging
import time
import openai
from ai_app1.service.multiply import aiTools, TOOL_FUNCTIONS

ai_client_logger = logging.getLogger("ai_client")


class AiClient:
    """
    MiniMax-M2.7 模型客户端。

    Attributes:
        ai_api_key: API 密钥（初始化时脱敏，仅显示前 8 位）
        client: openai.OpenAI 实例，复用 HTTP 连接池
    """

    def __init__(self, ai_api_key: str):
        self.ai_api_key = ai_api_key[:8] + "****"  # 脱敏日志输出
        self.client = openai.OpenAI(
            base_url="https://api.minimaxi.com/v1",
            api_key=ai_api_key
        )
        ai_client_logger.info(f"AiClient 初始化: base_url=https://api.minimaxi.com/v1")

    async def chat(self, messages: list, use_tools: bool = False) -> str:
        """
        发起一次对话请求（同步工具调用后的最终响应）。

        流程：
            1. 若 use_tools=True，附加工具定义发送首轮请求
            2. 若返回 tool_calls，执行工具并追加结果，再次请求获取最终文本
            3. 若无 tool_calls，直接返回 content

        Args:
            messages: 完整的消息列表（含 system/user/assistant/tool 各角色）
            use_tools: 是否启用工具调用（目前仅 summarize 使用 False）

        Returns:
            AI 生成的文本回复

        Raises:
            openai API 相关异常（由调用方统一处理）
        """
        step_id = id(messages)
        ai_client_logger.debug(f"[{step_id}] chat 调用: use_tools={use_tools}, messages={len(messages)}")
        start = time.monotonic()

        try:
            kwargs = {
                "model": "MiniMax-M2.7",
                "messages": messages
            }
            if use_tools:
                kwargs["tools"] = aiTools
                ai_client_logger.debug(f"[{step_id}] 启用 tools，tools 数量: {len(aiTools)}")

            # 首轮请求
            response = self.client.chat.completions.create(**kwargs)
            response_message = response.choices[0].message
            ai_client_logger.debug(
                f"[{step_id}] 首轮响应: content_len={len(response_message.content or '')}, "
                f"tool_calls={getattr(response_message, 'tool_calls', None) is not None}"
            )

            if not use_tools:
                ai_client_logger.debug(f"[{step_id}] chat 完成（无工具）: 耗时={time.monotonic() - start:.2f}s")
                return response_message.content or ""

            if not response_message.tool_calls:
                ai_client_logger.debug(f"[{step_id}] 无 tool_calls，直接返回文本")
                return response_message.content or ""

            # 存在工具调用：逐个执行工具，收集结果
            ai_client_logger.info(f"[{step_id}] 检测到 tool_calls: {len(response_message.tool_calls)}")
            for tc in response_message.tool_calls:
                ai_client_logger.debug(f"  - 函数: {tc.function.name}, 参数: {tc.function.arguments}")

            tool_results = []
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in TOOL_FUNCTIONS:
                    try:
                        result = TOOL_FUNCTIONS[func_name](**func_args)
                        ai_client_logger.debug(f"[{step_id}] 工具执行成功: {func_name}({func_args}) = {result}")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        ai_client_logger.error(f"[{step_id}] 工具执行异常: {func_name}, error={e}")
                else:
                    result = f"Unknown tool: {func_name}"
                    ai_client_logger.warning(f"[{step_id}] 未知工具: {func_name}")

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(result)
                })

            # 将工具调用消息和结果追加到 messages，再次请求获取最终回复
            messages.append(response_message.model_dump())
            messages.extend(tool_results)
            ai_client_logger.debug(f"[{step_id}] 工具结果追加: {len(tool_results)} 条")

            response = self.client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=messages
            )
            final_content = response.choices[0].message.content or ""
            ai_client_logger.info(
                f"[{step_id}] chat 最终响应: content_len={len(final_content)}, "
                f"耗时={time.monotonic() - start:.2f}s"
            )
            return final_content

        except Exception as e:
            ai_client_logger.error(f"[{step_id}] chat 异常: {type(e).__name__}: {e}")
            raise

    async def summarize(self, history: list) -> str:
        """
        将对话历史压缩为一段摘要文本。

        通过将 history 作为 user 消息内容发送给 LLM，让其总结关键信息，
        用于后续对话的上下文复用，减少 token 消耗。

        Args:
            history: 当前会话的 history 列表（不含 system prompt）

        Returns:
            LLM 生成的一段摘要文字
        """
        ai_client_logger.debug(f"summarize 调用: history_len={len(history)}")
        start = time.monotonic()

        prompt = [
            {"role": "user", "content": "请总结以下对话的关键信息，用于后续对话参考"},
            {"role": "user", "content": str(history)}
        ]

        result = await self.chat(prompt, use_tools=False)
        ai_client_logger.info(f"summarize 完成: result_len={len(result)}, 耗时={time.monotonic() - start:.2f}s")
        return result

    async def run_agent(self, messages: list) -> str:
        """
        工具增强型对话入口，内部自动处理多轮工具调用循环。

        流程（最多 MAX_STEPS = 10 轮）：
            1. 发送消息 + tools 定义给模型
            2. 若模型返回 tool_calls，执行对应函数，追加结果，循环
            3. 若无 tool_calls，返回文本内容

        与 chat() 的区别：
        - chat() 仅处理一轮工具调用，用于 summarize 等简单场景
        - run_agent() 处理多轮，用于复杂任务（可能多次调用工具）

        Args:
            messages: 完整的消息列表（含 system/user/assistant）

        Returns:
            AI 最终文本回复，或 "Agent stopped: max steps reached"（达到步数上限）
        """
        run_id = id(messages)
        ai_client_logger.info(f"[{run_id}] run_agent 启动: messages={len(messages)}")
        MAX_STEPS = 10

        for step in range(MAX_STEPS):
            ai_client_logger.debug(f"[{run_id}] Step {step + 1}/{MAX_STEPS}: 发送请求")
            start = time.monotonic()

            response = self.client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=messages,
                tools=aiTools,
            )

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            ai_client_logger.debug(
                f"[{run_id}] Step {step + 1} 响应: tool_calls={'有' if tool_calls else '无'}, "
                f"content_len={len(msg.content or '')}, 耗时={time.monotonic() - start:.2f}s"
            )

            if not tool_calls:
                ai_client_logger.info(f"[{run_id}] run_agent 结束（无 tool_calls）: step={step + 1}")
                return msg.content or ""

            ai_client_logger.info(f"[{run_id}] Step {step + 1} 工具调用: {len(tool_calls)} 个")

            # 执行工具并收集结果
            tool_results = []
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                try:
                    if func_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[func_name](**func_args)
                        ai_client_logger.debug(f"[{run_id}] 工具执行: {func_name}({func_args}) = {result}")
                    else:
                        result = f"Unknown tool: {func_name}"
                        ai_client_logger.warning(f"[{run_id}] 未知工具: {func_name}")
                except Exception as e:
                    result = f"Error: {str(e)}"
                    ai_client_logger.error(f"[{run_id}] 工具异常: {func_name}, error={e}")

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(result),
                })

            messages.append(msg.model_dump())
            messages.extend(tool_results)
            ai_client_logger.debug(f"[{run_id}] Step {step + 1} 工具结果追加: {len(tool_results)} 条")

        # 达到最大步数限制仍未返回（非工具调用场景）
        ai_client_logger.warning(f"[{run_id}] run_agent 达到最大步数限制: {MAX_STEPS}")
        return "Agent stopped: max steps reached"