"""
Session Manager — 会话生命周期 + 消息构建 + 摘要管理

替代原 session.py 中的过程式函数，面向对象封装。
集成 FailureCollector，自动收集低置信度和未命中样本。
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from rag_framework.core.config import RAGSettings
from rag_framework.core.logger import session_logger
from rag_framework.domain.base import DomainPlugin, QueryRoute
from rag_framework.session.base import SessionStore, SessionData

if TYPE_CHECKING:
    from rag_framework.llm.base import LLMClient
    from rag_framework.retrieval.base import Retriever
    from rag_framework.retrieval.query_rewriter.base import QueryRewriter


class SessionManager:
    """
    会话管理器。

    负责：
    - 会话 CRUD
    - 消息历史维护
    - Token 预算监控
    - 自动摘要
    - 构建带检索上下文的 messages
    """

    def __init__(
        self,
        store: SessionStore,
        llm: LLMClient,
        retriever: Retriever,
        domain: DomainPlugin,
        settings: RAGSettings | None = None,
        rule_rewriter: QueryRewriter | None = None,
        llm_rewriter: QueryRewriter | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._retriever = retriever
        self._domain = domain
        self._settings = settings or RAGSettings()
        self._rule_rewriter = rule_rewriter
        self._llm_rewriter = llm_rewriter

    def get_session(self, user_id: str) -> SessionData:
        """获取或创建会话。"""
        return self._store.get(user_id)

    def add_user_message(self, session: SessionData, message: str) -> None:
        """添加用户消息。"""
        session_logger.debug(f"用户消息入栈: history_len={len(session.history)}")
        session.history.append({"role": "user", "content": message})

    def add_assistant_message(self, session: SessionData, message: str) -> None:
        """添加助手消息。"""
        session_logger.debug(f"助手消息入栈: history_len={len(session.history)}")
        session.history.append({"role": "assistant", "content": message})

    def estimate_tokens(self, messages: list[dict]) -> int:
        """估算消息 token 数。"""
        total = 0
        for m in messages:
            text = m.get("content", "")
            total += self._domain.estimate_tokens(text)
        return total

    def should_summarize(self, session: SessionData) -> bool:
        """判断是否需要触发摘要压缩。"""
        raw = self._build_raw_messages(session)
        total = self.estimate_tokens(raw)
        result = total >= session.token_budget
        if result:
            session_logger.info(
                f"触发 summarize: total_tokens={total}, budget={session.token_budget}"
            )
        return result

    async def summarize(self, session: SessionData) -> str:
        """生成对话摘要。"""
        session_logger.info(f"开始 summarize: history_len={len(session.history)}")
        start = time.monotonic()
        summary = await self._llm.summarize(session.history)
        session.summary = summary
        session_logger.info(
            f"summarize 完成: result_len={len(summary)}, 耗时={time.monotonic()-start:.2f}s"
        )
        return summary

    def trim_history(self, session: SessionData) -> None:
        """裁剪历史到 max_history 条。"""
        max_h = self._settings.max_history
        if len(session.history) <= max_h:
            return
        trimmed_count = len(session.history) - max_h
        session.trimmed = session.history[:trimmed_count]
        session.history = session.history[trimmed_count:]
        session_logger.info(
            f"裁剪历史: 裁掉 {trimmed_count} 条, 剩余 {len(session.history)} 条"
        )

    def _build_routes(self, query: str, history: list[dict]) -> list[QueryRoute]:
        """
        根据 DomainPlugin 的分级规则生成多路检索路由。

        Returns:
            list[QueryRoute]，第一条为原始/改写后的主 query。
        """
        level = self._domain.rewrite_router_rules(query, history)
        session_logger.info(f"Rewrite level={level}: query={query!r}")

        if level == 2 and self._llm_rewriter is not None:
            return self._llm_rewriter.rewrite(query, history)

        if level == 1 and self._rule_rewriter is not None:
            return self._rule_rewriter.rewrite(query, history)

        # Level 0 / None：只做查询分类，不扩写
        route = self._domain.classify_query(query, history)
        session_logger.info(f"Rewrite level={level}: 跳过改写, route_type={route.type!r}")
        return [route]

    async def build_messages(self, session: SessionData, req_msg: str) -> list[dict]:
        """
        构建完整 messages（含检索上下文）。

        顺序：system → summary → history → [token 预警] → retrieved context
        retrieve() 为 async，在此 await 不阻塞事件循环。

        集成 FailureCollector：自动收集低置信度和未命中样本。
        """
        messages = self._build_raw_messages(session)

        if self.should_summarize(session):
            messages.append({
                "role": "user",
                "content": "【注意】对话即将超出上下文限制，请先简洁总结之前的关键信息，再继续回答。"
            })

        # 1. 分级 rewrite + classify → 多路 QueryRoute
        t_rewrite_start = time.perf_counter()
        routes = self._build_routes(req_msg, session.history)
        t_rewrite = (time.perf_counter() - t_rewrite_start) * 1000
        session_logger.info(f"检索路由: routes={[r.type for r in routes]}")

        # 2. 异步多路检索
        result = await self._retriever.retrieve(routes)

        # ── 打印检索到的片段内容 ──────────────────────────────────────────────
        if result.docs:
            for idx, doc in enumerate(result.docs):
                session_logger.info(
                    f"[检索片段 {idx}] id={doc.id!r} source={doc.source!r} "
                    f"score={doc.score:.4f}\n{doc.text}"
                )
        else:
            session_logger.info("[检索片段] 未检索到任何文档")
        # ─────────────────────────────────────────────────────────────────────

        context = "\n\n".join(d.text for d in result.docs) if result.docs else ""
        top_ce = result.metadata.get("top_ce", 0.0)
        n_chunks = len(result.docs)
        threshold = self._settings.low_confidence_threshold

        # ── Failure Analysis 收集 ─────────────────────────────────────────────
        from rag_framework.eval.failure_analysis import get_failure_collector
        collector = get_failure_collector()
        trace_dict = result.metadata.get("trace", {})
        trace_dict["rewrite_latency_ms"] = t_rewrite

        if not result.docs:
            # 未检索到任何文档
            collector.collect_miss(
                query=req_msg,
                trace=trace_dict,
                session_id=session.user_id,
            )
            session_logger.warning("未检索到任何文档")
            messages.append({
                "role": "user",
                "content": self._domain.fallback_response("no_results")
            })
        elif top_ce < threshold:
            # 低置信度
            collector.collect_low_ce(
                query=req_msg,
                top_ce=top_ce,
                trace=trace_dict,
                session_id=session.user_id,
            )
            session_logger.warning(f"低置信度检索 (top_ce={top_ce:.3f})，触发拒答")
            messages.append({
                "role": "user",
                "content": self._domain.fallback_response("low_confidence")
            })
        else:
            messages.append({"role": "user", "content": f"参考资料：{context}"})
            session_logger.info(f"追加参考资料 ({n_chunks} 片段, top_ce={top_ce:.3f})")

        return messages

    def _build_raw_messages(self, session: SessionData) -> list[dict]:
        """构建不含检索上下文的 messages。"""
        messages = [{"role": "system", "content": self._domain.system_prompt}]
        if session.summary:
            messages.append({
                "role": "user",
                "content": f"【历史摘要】{session.summary}"
            })
        messages.extend(session.history)
        return messages

    async def chat_stream(self, query: str, user_id: str = "default_user"):
        """端到端流式对话（生成器）。"""
        session = self.get_session(user_id)
        self.add_user_message(session, query)

        messages = await self.build_messages(session, query)
        full_reply = ""

        async for chunk in self._llm.chat_stream(messages, use_tools=False):
            full_reply += chunk
            yield chunk

        # 流结束后维护 session
        self.add_assistant_message(session, full_reply)
        if self.should_summarize(session):
            try:
                await self.summarize(session)
            except Exception as e:
                session_logger.error(f"summarize 失败: {e}")
        self.trim_history(session)
        self._store.save(session)
