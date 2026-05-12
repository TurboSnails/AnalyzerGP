"""
AiClient：封装与主答案 LLM 的交互逻辑，支持多后端切换。

提供两个核心接口：
- chat(messages, use_tools=False)：普通对话，返回文本
- run_agent(messages)：工具增强对话，内部自动处理工具调用循环

后端通过 LLM_BACKEND 环境变量切换：
  minimax  → MiniMax-M2.7 远程（生产）
  ollama   → 本地 Ollama OpenAI 兼容端点（开发）
  openai   → GPT/DeepSeek/通义等任意 OpenAI 兼容厂商
所有后端走同一份 openai SDK 调用，零代码差异。
"""
import json
import logging
import os
import time
import openai
from ai_app1.core.config import LLM_BACKEND, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
from ai_app1.service.tools import aiTools, TOOL_FUNCTIONS

ai_client_logger = logging.getLogger("ai_client")

# 限制单次回答的最大 token 数。RAG 场景 200-400 字够用，避免 LLM 长篇大论拖慢响应。
# 设 0 = 不限制（原行为）。MiniMax 默认无限制时容易输出 800+ 字，浪费 4-6s 生成时间。
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))


class AiClient:
    """
    多后端 LLM 客户端（OpenAI 兼容协议）。

    Attributes:
        ai_api_key: API 密钥（初始化时脱敏，仅显示前 8 位）
        client: openai.AsyncOpenAI 实例，复用 HTTP 连接池
        _model: 当前使用的模型名（按后端解析）
        _backend: 当前后端标识（用于日志）
    """

    def __init__(self, ai_api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None):
        api_key = ai_api_key or LLM_API_KEY or "placeholder"
        self.ai_api_key = (api_key[:8] + "****") if api_key else "none"
        self._backend = LLM_BACKEND
        self._model = model or LLM_MODEL
        self._base_url = base_url or LLM_BASE_URL
        self.client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key,
        )
        ai_client_logger.info(
            f"AiClient 初始化: backend={self._backend}, "
            f"base_url={self._base_url}, model={self._model}"
        )

    # ─── 流式对话 ───────────────────────────────────────────────

    async def _stream_response(self, messages: list, use_tools: bool = False):
        """
        流式响应：每个 token 立即 yield，不缓冲。
        use_tools 只控制是否将工具定义发给模型，tool_calls 由 stream_run_agent 负责处理。
        """
        kwargs = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if LLM_MAX_TOKENS > 0:
            kwargs["max_tokens"] = LLM_MAX_TOKENS
        if use_tools:
            kwargs["tools"] = aiTools

        t0 = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)

        first_token_ms = None
        char_count = 0
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                char_count += len(delta.content)
                yield delta.content

        total_ms = (time.perf_counter() - t0) * 1000
        gen_ms = total_ms - (first_token_ms or 0)
        chars_per_sec = (char_count / gen_ms * 1000) if gen_ms > 0 else 0
        ai_client_logger.info(
            f"[{self._backend}] LLM 完成: TTFT={first_token_ms:.0f}ms, "
            f"生成={gen_ms:.0f}ms, 总={total_ms:.0f}ms, "
            f"输出={char_count}字符 ({chars_per_sec:.0f}字符/秒), "
            f"max_tokens={LLM_MAX_TOKENS}"
        )

    async def stream_chat(self, messages: list, use_tools: bool = False):
        """
        流式发起一次对话请求，yield 每个 token chunk。

        工具调用场景会先完整收集 tool_calls，再执行工具，最后继续流式返回最终响应。
        """
        async for token in self._stream_response(messages, use_tools):
            yield token

    async def stream_run_agent(self, messages: list):
        """
        流式工具增强型对话入口，内部自动处理多轮工具调用循环。
        每轮在流式响应中直接收集 tool_calls，避免额外非流式请求导致的延迟。
        """
        MAX_STEPS = 10

        for step in range(MAX_STEPS):
            full = ""
            tool_calls: list[dict] = []

            stream_kwargs = {
                "model": self._model,
                "messages": messages,
                "tools": aiTools,
                "stream": True,
            }
            if LLM_MAX_TOKENS > 0:
                # 双保险：max_tokens（OpenAI 旧式）+ max_completion_tokens（新式）
                # MiniMax 部分模型只认其中一个，全发避免猜参数名
                stream_kwargs["max_tokens"] = LLM_MAX_TOKENS
                stream_kwargs["extra_body"] = {
                    "max_completion_tokens": LLM_MAX_TOKENS,
                    "tokens_to_generate": LLM_MAX_TOKENS,  # MiniMax 文档曾用此名
                }

            t0 = time.perf_counter()
            response = await self.client.chat.completions.create(**stream_kwargs)
            first_token_ms: float | None = None

            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    full += delta.content
                    yield delta.content

                # 增量收集 tool_calls（流式中直接获取，省去第二次请求）
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}, "type": "function"})
                        if tc.id:
                            tool_calls[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments

            total_ms = (time.perf_counter() - t0) * 1000
            gen_ms = total_ms - (first_token_ms or 0)
            chars_per_sec = (len(full) / gen_ms * 1000) if gen_ms > 0 else 0
            prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            ai_client_logger.info(
                f"[{self._backend}] LLM step={step+1}: "
                f"prompt={prompt_chars}字符, "
                f"TTFT={first_token_ms or 0:.0f}ms, "
                f"生成={gen_ms:.0f}ms, "
                f"总={total_ms:.0f}ms, "
                f"输出={len(full)}字符 ({chars_per_sec:.0f}字符/秒), "
                f"tool_calls={len(tool_calls)}, "
                f"max_tokens={LLM_MAX_TOKENS}"
            )

            if not full and not tool_calls:
                break

            if not tool_calls:
                break

            tool_results = []
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                try:
                    result = TOOL_FUNCTIONS[func_name](**func_args) if func_name in TOOL_FUNCTIONS else f"Unknown tool: {func_name}"
                except Exception as e:
                    result = f"Error: {str(e)}"
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": str(result)
                })

            messages.append({
                "role": "assistant",
                "content": full,
                "tool_calls": tool_calls
            })
            messages.extend(tool_results)

    # ─── 非流式对话（兼容保留） ─────────────────────────────────

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
                "model": self._model,
                "messages": messages,
            }
            if LLM_MAX_TOKENS > 0:
                kwargs["max_tokens"] = LLM_MAX_TOKENS
            if use_tools:
                kwargs["tools"] = aiTools
                ai_client_logger.debug(f"[{step_id}] 启用 tools，tools 数量: {len(aiTools)}")

            # 首轮请求
            response = await self.client.chat.completions.create(**kwargs)
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

            final_kwargs = {"model": self._model, "messages": messages}
            if LLM_MAX_TOKENS > 0:
                final_kwargs["max_tokens"] = LLM_MAX_TOKENS
            response = await self.client.chat.completions.create(**final_kwargs)
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

        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in history
        )
        prompt = [
            {"role": "user", "content": "请总结以下对话的关键信息，用于后续对话参考"},
            {"role": "user", "content": history_text}
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

            agent_kwargs = {
                "model": self._model,
                "messages": messages,
                "tools": aiTools,
            }
            if LLM_MAX_TOKENS > 0:
                agent_kwargs["max_tokens"] = LLM_MAX_TOKENS
            response = await self.client.chat.completions.create(**agent_kwargs)

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