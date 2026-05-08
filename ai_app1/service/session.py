"""
会话管理模块：维护用户对话历史、摘要压缩和文档检索。

设计思路：
- 每个 user_id 对应一个 SessionData（内存字典，进程重启后丢失）
- history 仅保留最近 MAX_HISTORY 条，超出部分裁剪到 trimmed（防止丢失）
- token 预算耗尽时触发 summarize，将历史压缩为 summary 供后续上下文复用
- retrieve_doc 解析 docs.txt，根据用户 query 匹配错误类型并返回解答
"""
from typing import TypedDict

from ai_app1.core.logger import session_logger, retrieve_doc_logger
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

    采用字符数 / 4 的粗略估算（中文语境下误差可接受），
    优于每次都调用 tiktoken（增加依赖和耗时）。
    """
    tokens = sum(len(m.get("content", "")) // 4 for m in messages)
    session_logger.debug(f"预估 token 数: {tokens}")
    return tokens


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

    包含：raw messages + 压缩提示（若 token 超预算）+ docs.txt 检索结果

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
    if context:
        messages.append({"role": "user", "content": f"参考资料：{context}"})
        session_logger.info(f"追加参考资料: {context[:50]}...")
    else:
        session_logger.debug(f"未检索到相关文档: {req_msg[:30]}")

    session_logger.info(f"构建最终 messages: {len(messages)} 条")
    return messages


def retrieve_doc(query: str):
    """
    从 docs.txt 检索与 query 相关的错误解答。

    docs.txt 格式（以空行分隔条目）：
        1. NullPointerException：空指针
        解决：检查对象是否初始化

    Args:
        query: 用户输入的原始消息

    Returns:
        匹配到的错误解答；若未命中则返回 None
    """
    retrieve_doc_logger.debug(f"检索文档: query={query}")

    # 动态计算 docs.txt 路径（相对于 session.py 所在目录）
    import os
    import re
    docs_path = os.path.join(os.path.dirname(__file__), "..", "docs.txt")
    if not os.path.exists(docs_path):
        retrieve_doc_logger.warning(f"docs.txt 不存在: {docs_path}")
        return None

    with open(docs_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析：按空行分块，每块第一行匹配 "序号. 错误名：描述"，
    # 紧随的 "解决：" 行作为解答，同时建立大小写不敏感的 key
    entries = {}
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\d+\.\s*(.+?)：(.*)$", line)
            if m:
                error_name = m.group(1).strip()
                error_desc = m.group(2).strip()
                # 向下查找 "解决：" 开头的行
                solution = ""
                for j in range(i + 1, len(lines)):
                    sol_line = lines[j].strip()
                    if sol_line.startswith("解决："):
                        solution = sol_line.replace("解决：", "").strip()
                        break
                if solution:
                    entries[error_name] = f"{error_desc}。{solution}"
                    entries[error_name.lower()] = f"{error_desc}。{solution}"

    query_lower = query.lower()
    for error_name, answer in entries.items():
        if error_name in query_lower or error_name in query:
            retrieve_doc_logger.info(f"命中文档条目: error_name={error_name}")
            return answer

    retrieve_doc_logger.debug(f"未命中任何文档条目")
    return None