from typing import TYPE_CHECKING
from langchain_core.messages import SystemMessage, BaseMessage # BaseMessage 추가
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini LLM 추가

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    print(f"\n--- Summary_Module 실행 ---")
    messages: list[BaseMessage] = state.get("messages", [])
    session_id = state.get("session_id")

    print(f"Summary_Module이 Gemini로 대화 요약 중입니다.")

    # # Gemini LLM 준비
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash-lite",
    #     temperature=0.3,
    #     google_api_key=GEMINI_API_KEY,
    #     max_retries=2
    # )

    # # 요약 프롬프트 추가
    # summary_prompt = SystemMessage(content="아래의 대화 전체를 한국어로 간결하게 요약해줘. 중요한 정보와 맥락을 빠짐없이 포함해줘.\n")
    # llm_input = [summary_prompt] + messages

    # # Gemini LLM 호출
    # summary_response = await llm.ainvoke(llm_input)
    # summary_text = summary_response.content
    # print(f"요약: {summary_text}")

    summary_text = "요약 테스트"

    updated_messages = messages + [SystemMessage(content=f"요약 생성됨: {summary_text}")]
    return {"summary": summary_text, "messages": updated_messages} 