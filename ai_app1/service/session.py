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


def build_messages(session: SessionData, reqMsg: str) -> list:
    messages = build_messages_raw(session)
    if should_summarize(session):
        messages.append({
            "role": "user",
            "content": "【注意】对话即将超出上下文限制，请先简洁总结之前的关键信息，再继续回答。"
        })
    context = retrieve_doc(reqMsg)
    if context:
        messages.append({
            "role": "user",
            "content": f"参考资料：{context}"
        })

    return messages


def retrieve_doc(query: str):
    import os
    import re

    docs_path = os.path.join(os.path.dirname(__file__), "..", "docs.txt")
    if not os.path.exists(docs_path):
        return "未找到相关文档，请描述具体的错误信息"

    with open(docs_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 格式：
    # 1. NullPointerException：空指针
    # 解决：检查对象是否初始化
    entries = {}
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\d+\.\s*(.+?)：(.*)$", line)
            if m:
                error_name = m.group(1).strip()
                error_desc = m.group(2).strip()
                # 找对应的解决行
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
            return answer

    return None
