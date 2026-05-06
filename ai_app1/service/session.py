from typing import TypedDict

class SessionData(TypedDict):
    history: list
    summary: str
    trimmed: list
    token_budget: int  # 本轮可用token预算（动态）


MAX_HISTORY = 4
SYSTEM_PROMPT = "你是一个专业的Android开发助手，回答要简洁、准确"
DEFAULT_TOKEN_BUDGET = 4096  # MiniMax-M2 模型 context 的一半左右


user_sessions: dict[str, SessionData] = {}


def get_session(user_id: str) -> SessionData:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "summary": "",
            "trimmed": [],
            "token_budget": DEFAULT_TOKEN_BUDGET,
        }
    return user_sessions[user_id]


def add_user_message(session: SessionData, message: str):
    session["history"].append({"role": "user", "content": message})


def add_assistant_message(session: SessionData, message: str):
    session["history"].append({"role": "assistant", "content": message})


def estimate_tokens(messages: list) -> int:
    return sum(len(m.get("content", "")) // 4 for m in messages)


def should_summarize(session: SessionData) -> bool:
    messages = build_messages_raw(session)
    total_tokens = estimate_tokens(messages)
    return total_tokens >= session["token_budget"]


def update_summary(session: SessionData, summary: str):
    session["summary"] = summary


def trim_history(session: SessionData):
    if len(session["history"]) <= MAX_HISTORY:
        return

    trimmed_count = len(session["history"]) - MAX_HISTORY
    trimmed_msgs = session["history"][:trimmed_count]
    session["trimmed"] = trimmed_msgs
    session["history"] = session["history"][trimmed_count:]


def build_messages_raw(session: SessionData) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if session["summary"]:
        messages.append({"role": "user", "content": f"【历史摘要】{session['summary']}"})
    messages.extend(session["history"])
    return messages


def build_messages(session: SessionData) -> list:
    messages = build_messages_raw(session)
    if should_summarize(session):
        messages.append({
            "role": "user",
            "content": "【注意】对话即将超出上下文限制，请先简洁总结之前的关键信息，再继续回答。"
        })
    return messages