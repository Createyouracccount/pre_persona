import operator
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Sequence[BaseMessage] # Annotated 및 operator.add 제거
    user_input: str                                  # 현재 사용자 입력
    user_query: str | None                           # LLM 입력용 사용자 쿼리
    user_id: str | None
    session_id: str | None
    retrieved_user_info: dict | None
    retrieved_status_info: dict | None
    summary: str | None
    context: list | None                             # RAG 검색 결과
    error_message: str | None 