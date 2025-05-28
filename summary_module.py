from typing import TYPE_CHECKING
from langchain_core.messages import SystemMessage, BaseMessage # BaseMessage 추가

if TYPE_CHECKING:
    # AgentState의 실제 정의는 main_1.py에 있으므로 순환 참조 방지를 위해 TYPE_CHECKING 사용
    from typing import TypedDict, Annotated, Sequence
    import operator

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_input: str
        user_id: str | None
        session_id: str | None
        retrieved_user_info: dict | None
        retrieved_status_info: dict | None
        summary: str | None
        error_message: str | None

async def summary_module_function(state: "AgentState"):
    print(f"\\n--- Summary_Module 실행 ---")
    # state.get("messages", [])의 반환 타입이 List[BaseMessage]임을 명시 (실제 AgentState 정의와 일치)
    messages: list[BaseMessage] = state.get("messages", [])
    
    print(f"Summary_Module이 대화를 요약 중입니다.")
    session_id = state.get("session_id")
    summary_text = f"대화 요약 완료 (총 {len(messages)}개의 메시지 기반). 세션 ID: {session_id}"
    print(f"요약: {summary_text}")
    
    # 기존 메시지 리스트에 새로운 SystemMessage를 추가하여 반환
    updated_messages = messages + [SystemMessage(content=f"요약 생성됨: {summary_text}")]
    
    return {"summary": summary_text, "messages": updated_messages} 