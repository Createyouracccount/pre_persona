import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # 대화 기록용
    user_input: str                                  # 현재 사용자 입력
    user_id: str | None
    session_id: str | None
    retrieved_user_info: dict | None
    retrieved_status_info: dict | None
    summary: str | None
    error_message: str | None 