import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from LLM_module.utils.mcp_tools import tools
from dotenv import load_dotenv

from LLM_module.utils.state import AgentState

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 에이전트별 모델 설정을 위한 딕셔너리
AGENT_MODEL_CONFIG = {
    "main_agent": "gemini-2.0-flash-lite",
    "greeting_agent": "gemini-2.0-flash-lite",  # 예시: 필요시 모델 변경
    "filler_agent": "gemini-2.0-flash-lite",    # 예시: 필요시 모델 변경
    "summary_module": "gemini-2.0-flash-lite",
    "rag_agent": "gemini-2.0-flash-lite"        # 예시: 필요시 모델 변경
}

# AgentState 타입 힌트는 agent.py에서 import해서 사용하므로 여기서는 타입 체크만 참고

async def main_agent_function(state: AgentState):
    print(f"\n--- 메인 LLM 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))

    if user_input:
        current_messages.append(HumanMessage(content=user_input))
        print(f"메인 LLM 에이전트가 사용자 입력을 받았습니다: '{user_input}'")

    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["main_agent"],
        temperature=0.7,
        google_api_key=GEMINI_API_KEY, # 직접 GEMINI_API_KEY 변수 사용
        max_retries=2
    )
    llm_with_tools = llm.bind_tools(tools)
    # 비동기 함수이므로 agent.py에서 await로 호출해야 함
    llm_response_message = await llm_with_tools.ainvoke(current_messages)
    return {
        "messages": current_messages + [llm_response_message],
        "user_input": ""
    }

async def greeting_agent_function(state: AgentState):
    print(f"\n--- 인사 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))
    # llm = ChatGoogleGenerativeAI(
    #     model=AGENT_MODEL_CONFIG["greeting_agent"], # 설정된 모델 사용
    #     temperature=0.7,
    #     google_api_key=GEMINI_API_KEY, # 직접 GEMINI_API_KEY 변수 사용
    #     max_retries=2
    # )
    # Note: The LLM initialization above was commented out as it appeared unused.
    # It has been updated to use the AGENT_MODEL_CONFIG.
    # Please review if it's needed or can be removed.
    return {
        "messages": current_messages,
        "user_input": ""
    }

async def filler_agent_function(state: AgentState):
    print(f"\n--- 시간벌기용 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))
    # 이 에이전트가 LLM을 사용한다면 아래와 같이 설정할 수 있습니다.
    # llm = ChatGoogleGenerativeAI(
    #     model=AGENT_MODEL_CONFIG["filler_agent"],
    #     temperature=0.7,
    #     google_api_key=GEMINI_API_KEY,
    #     max_retries=2
    # )
    return {
        "messages": current_messages,
        "user_input": ""
    }

async def summary_module_function(state: AgentState):
    print(f"\n--- Summary_Module 실행 ---")
    messages: list[BaseMessage] = state.get("messages", [])
    session_id = state.get("session_id")

    print(f"Summary_Module이 Gemini로 대화 요약 중입니다.")

    # Gemini LLM 준비
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["summary_module"],
        temperature=0.3,
        google_api_key=GEMINI_API_KEY, # 직접 GEMINI_API_KEY 변수 사용
        max_retries=2
    )

    # 요약 프롬프트 추가
    summary_prompt = SystemMessage(content="아래의 대화 전체를 한국어로 간결하게 요약해줘. 중요한 정보와 맥락을 빠짐없이 포함해줘.\n")
    llm_input = [summary_prompt] + messages

    # Gemini LLM 호출
    summary_response = await llm.ainvoke(llm_input)
    summary_text = summary_response.content
    print(f"요약: {summary_text}")

    updated_messages = messages + [SystemMessage(content=f"요약 생성됨: {summary_text}")]
    return {"summary": summary_text, "messages": updated_messages} 


async def rag_function(state: AgentState):
    print(f"\n--- RAG 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))
    # 이 에이전트가 LLM을 사용한다면 아래와 같이 설정할 수 있습니다.
    # llm = ChatGoogleGenerativeAI(
    #     model=AGENT_MODEL_CONFIG["rag_agent"],
    #     temperature=0.7,
    #     google_api_key=GEMINI_API_KEY,
    #     max_retries=2
    # )
    return {
        "messages": current_messages,
        "user_input": ""
    }