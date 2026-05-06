from typing import TypedDict

class SessionData(TypedDict):
    history: list
    summary: str


MAX_HISTORY = 6
SYSTEM_PROMPT = "你是一个专业的Android开发助手，回答要简洁、准确"

user_sessions: dict[str, SessionData] = {}


def get_session(user_id: str) -> SessionData:
    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": [], "summary": ""}
    return user_sessions[user_id]


def add_user_message(session: SessionData, message: str):
    session["history"].append({"role": "user", "content": message})


def add_assistant_message(session: SessionData, message: str):
    session["history"].append({"role": "assistant", "content": message})


def update_summary(session: SessionData, summary: str):
    session["summary"] = summary


def trim_history(session: SessionData):
    session["history"] = session["history"][-MAX_HISTORY:]


def build_messages(session: SessionData) -> list:
    messages = [{"role": "user", "content": SYSTEM_PROMPT}]

    if session["summary"]:
        messages.append({
            "role": "user",
            "content": f"历史对话摘要：{session['summary']}"
        })

    messages.extend(session["history"][-MAX_HISTORY:])
    return messages