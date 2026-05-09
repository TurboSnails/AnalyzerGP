"""
会话管理模块：维护用户对话历史、摘要压缩和文档检索。

设计思路：
- 每个 user_id 对应一个 SessionData（内存字典，进程重启后丢失）
- history 仅保留最近 MAX_HISTORY 条，超出部分裁剪到 trimmed（防止丢失）
- token 预算耗尽时触发 summarize，将历史压缩为 summary 供后续上下文复用
"""
from typing import TypedDict

from ai_app1.core.logger import session_logger
from ai_app1.service.vector_store import query_db

class SessionData(TypedDict):
    """会话数据结构，所有字段均为进程内内存存储"""
    history: list       # 最近的对话记录 [{role, content}, ...]
    summary: str       # 历史对话摘要（token 预算耗尽时生成）
    trimmed: list       # 被裁剪的旧消息（暂时保留，不丢弃）
    token_budget: int  # 本会话剩余 token 预算


# ─── 配置常量 ───────────────────────────────────────────────
MAX_HISTORY = 4                  # 保留在 history 中的最近消息条数
SYSTEM_PROMPT = "你是一个专业的Android开发助手，回答要简洁、准确"
DEFAULT_TOKEN_BUDGET = 4096      # 约等于 MiniMax-M2 context 的一半


# ─── 会话存储 ───────────────────────────────────────────────
user_sessions: dict[str, SessionData] = {}


def get_session(user_id: str) -> SessionData:
    """
    获取或创建指定 user_id 的会话。

    Args:
        user_id: 用户唯一标识（目前固定为 "default_user"）

    Returns:
        该用户的 SessionData 实例
    """
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "summary": "",
            "trimmed": [],
            "token_budget": DEFAULT_TOKEN_BUDGET,
        }
        session_logger.info(f"创建新会话: user_id={user_id}")
    else:
        session_logger.debug(f"复用已有会话: user_id={user_id}, history_len={len(user_sessions[user_id]['history'])}")
    return user_sessions[user_id]


def add_user_message(session: SessionData, message: str):
    """将用户消息追加到历史记录"""
    session_logger.debug(f"用户消息入栈: history_len={len(session['history'])}")
    session["history"].append({"role": "user", "content": message})


def add_assistant_message(session: SessionData, message: str):
    """将 AI 助手回复追加到历史记录"""
    session_logger.debug(f"助手消息入栈: history_len={len(session['history'])}")
    session["history"].append({"role": "assistant", "content": message})


def estimate_tokens(messages: list) -> int:
    """
    估算一组消息的 token 总数。

    改进策略（应对 Android 开发场景的中英文混合）：
      - 中文字符（含全角标点）：~1.5 token/字
      - 英文/数字/代码符号/半角标点：~0.5 token/字符
    优于单纯 "字符数/4"（后者对中文严重低估，会导致过早触发 summarize）。
    """
    total = 0
    for m in messages:
        text = m.get("content", "")
        cn_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other_chars = len(text) - cn_chars
        total += int(cn_chars * 1.5 + other_chars * 0.5)
    session_logger.debug(f"预估 token 数: {total}")
    return total


def should_summarize(session: SessionData) -> bool:
    """
    判断当前会话是否需要触发摘要压缩。

    当 build_messages_raw 后的总 token 数超过 budget 时返回 True，
    触发条件发生在 AI 回复之后，避免 AI 还没看过新消息就被压缩。
    """
    messages = build_messages_raw(session)
    total_tokens = estimate_tokens(messages)
    result = total_tokens >= session["token_budget"]
    if result:
        session_logger.info(f"触发 summarize: total_tokens={total_tokens}, budget={session['token_budget']}")
    return result


def update_summary(session: SessionData, summary: str):
    """用 LLM 生成的新摘要替换旧摘要"""
    session_logger.info(f"更新摘要: summary_len={len(summary)}, history_len={len(session['history'])}")
    session["summary"] = summary


def trim_history(session: SessionData):
    """
    裁剪 history 到 MAX_HISTORY 条。

    被裁掉的旧消息存入 trimmed 而非丢弃，保留信息以防需要回溯。
    裁剪发生在 AI 回复之后，不会丢失本轮对话内容。
    """
    if len(session["history"]) <= MAX_HISTORY:
        session_logger.debug(f"无需裁剪: history_len={len(session['history'])} <= MAX_HISTORY={MAX_HISTORY}")
        return

    trimmed_count = len(session["history"]) - MAX_HISTORY
    trimmed_msgs = session["history"][:trimmed_count]
    session["trimmed"] = trimmed_msgs
    session["history"] = session["history"][trimmed_count:]
    session_logger.info(f"裁剪历史: 裁掉 {trimmed_count} 条, 剩余 {len(session['history'])} 条, 移到 trimmed {len(trimmed_msgs)} 条")


def build_messages_raw(session: SessionData) -> list:
    """
    构建发送给 LLM 的消息列表（不含文档检索结果）。

    顺序：system prompt → 历史摘要（若有）→ 最近 history
    注意：MiniMax 在 tool calling 模式下不支持 system 角色，因此用 role: user
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if session["summary"]:
        messages.append({"role": "user", "content": f"【历史摘要】{session['summary']}"})
    messages.extend(session["history"])
    session_logger.debug(f"构建 raw messages: {len(messages)} 条, history={len(session['history'])}, summary={'有' if session['summary'] else '无'}")
    return messages


def build_messages(session: SessionData, req_msg: str) -> list:
    """
    构建完整的发送给 run_agent 的 messages 列表。

    包含：raw messages + 压缩提示（若 token 超预算）+ 向量库检索结果

    Args:
        session: 当前会话
        req_msg: 用户原始请求消息，用于检索相关文档
    """
    messages = build_messages_raw(session)

    if should_summarize(session):
        session_logger.info("接近 token 上限，追加压缩提示")
        messages.append({
            "role": "user",
            "content": "【注意】对话即将超出上下文限制，请先简洁总结之前的关键信息，再继续回答。"
        })

    context = query_db(req_msg)
    print(f"---------------->query_db:${context}")
    if context:
        messages.append({"role": "user", "content": f"参考资料：{context}"})
        session_logger.info(f"追加参考资料: {context[:50]}...")
    else:
        session_logger.debug(f"未检索到相关文档: {req_msg[:30]}")

    session_logger.info(f"构建最终 messages: {len(messages)} 条")
    return messages

