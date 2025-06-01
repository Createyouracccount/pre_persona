import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from LLM_module.utils.mcp_tools import tools #테스트용 주석처리 테스트 끝나면 복구해야함
from LLM_module.utils.retriever_tool import retriever_tool
from LLM_module.utils.state import AgentState

# from state import AgentState

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 에이전트별 모델 설정을 위한 딕셔너리
AGENT_MODEL_CONFIG = {
    "main_agent": "gemini-2.0-flash-lite",
    "greeting_agent": "gemini-2.0-flash-lite",  # 예시: 필요시 모델 변경
    "filler_agent": "gemini-2.0-flash-lite",    # 예시: 필요시 모델 변경
    "summary_module": "gemini-2.0-flash-lite",
}

# 질문인지 판단하는 부분
class GreetingAnalysis(BaseModel):
    is_question: bool = Field(..., description="사용자의 입력이 명확한 질문이면 True, 단순 인사나 대화이면 False입니다.")
    answer: str = Field(..., description="사용자 입력에 대한 적절한 인사말 또는 답변입니다.")


# AgentState 타입 힌트는 agent.py에서 import해서 사용하므로 여기서는 타입 체크만 참고

async def main_agent_function(state: AgentState):
    print(f"\n--- 메인 LLM 에이전트 실행 ---")
    if state.get("user_query"):
        user_interaction_content = state.get("user_input", "")
    else:
        user_interaction_content = state.get("user_input", "")

    current_messages = list(state.get("messages", []))

    with open("prompts/main_agent_prompt.txt", "r") as file:
        main_agent_prompt = file.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", main_agent_prompt),
        ("user", "{user_interaction_content}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["main_agent"],
        temperature=0.7,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
    )
    chain = prompt | llm.bind_tools(tools)

    main_agent_response_message = await chain.ainvoke({"user_interaction_content": user_interaction_content})
    print(f"main_agent_function: {main_agent_response_message.content}")

    updated_messages = list(current_messages)
    updated_messages.append(main_agent_response_message)
    print(f"main_agent_function: ", state.get("messages"))
    return {
        "messages": updated_messages,
    }


async def greeting_agent_function(state: AgentState):
    print(f"\n--- 인사 에이전트 실행 ---")
    user_input_content = state.get("user_input", "")
    current_messages = list(state.get("messages", []))
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["greeting_agent"],
        temperature=0.7,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
    )

    with open("prompts/greeting_agent_prompt.txt", "r") as file:
        greeting_agent_prompt = file.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", greeting_agent_prompt),
        ("user", "{user_input_content}")
    ])
    structured_llm = llm.with_structured_output(GreetingAnalysis)
    chain = prompt | structured_llm
    analysis_result: GreetingAnalysis = await chain.ainvoke({"user_input_content": user_input_content})

    user_query_value = None
    updated_messages = list(current_messages)

    if analysis_result.is_question:
        print(f"--- 사용자 입력은 '질문'으로 판단됨 ---")
        user_query_value = user_input_content
    else:
        print(f"--- 사용자 입력은 '대화/인사'로 판단됨 ---")
        greeting_response_message = AIMessage(content=analysis_result.answer)
        print(f"greeting_agent_function (response): {greeting_response_message.content}")
        updated_messages.append(greeting_response_message)

    print(f"greeting_agent_function: ", state.get("messages"))
    print(f"DEBUG greeting_agent_function: messages TO BE RETURNED: {updated_messages}")
    return {
        "messages": updated_messages,
        "user_query": user_query_value,
    }


async def filler_agent_function(state: AgentState):
    print(f"\n--- 시간벌기용 에이전트 실행 ---")
    user_interaction_content = state.get("user_query", "")
    if not user_interaction_content:
        print("경고: filler_agent_function이 user_query 없이 호출되었습니다.")

    current_messages = list(state.get("messages", []))

    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["filler_agent"],
        temperature=0.7,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
    )

    with open("prompts/filler_agent_prompt.txt", "r") as file:
        filler_agent_prompt = file.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", filler_agent_prompt),
        ("user", "{user_interaction_content}")
    ])

    chain = prompt | llm
    filler_response = await chain.ainvoke({"user_interaction_content": user_interaction_content})
    print(f"filler_agent_function: {filler_response.content}")

    updated_messages = list(current_messages)
    updated_messages.append(filler_response)
    print(f"filler_agent_function: ", state.get("messages"))
    return {
        "messages": updated_messages,
    }


# def human_node(state):
#     user_input = interrupt("사람의 입력을 기다립니다. 입력해주세요.")
#     state["user_input"] = user_input
#     return state


async def summary_module_function(state: AgentState):
    print(f"\n--- Summary_Module 실행 ---")
    messages: list[BaseMessage] = state.get("messages", [])
    session_id = state.get("session_id")

    print(f"Summary_Module이 Gemini로 대화 요약 중입니다.")

    # Gemini LLM 준비
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL_CONFIG["summary_module"],
        temperature=0.3,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
    )

    # 요약 프롬프트 추가
    summary_prompt = SystemMessage(content="아래의 대화 전체를 한국어로 간결하게 요약해줘. 중요한 정보와 맥락을 빠짐없이 포함해줘.\n")
    llm_input = [summary_prompt] + messages

    # Gemini LLM 호출
    summary_response = await llm.ainvoke(llm_input)
    summary_text = summary_response.content
    print(f"요약: {summary_text}")

    return {"summary": summary_text} 


# async def rag_function(state: AgentState):
#     print(f"\n--- RAG 실행 ---")
#     state.search_results = retriever_tool(state.search_query)
#     return "generate"



if __name__ == "__main__":
    import asyncio
    from state import AgentState
    from mcp_tools import tools

    async def test_greeting_agent():
        # 테스트용 AgentState 생성
        state = AgentState(
            user_input="카드 신청하려면 어떻게 해야하나요",
            messages=[]
        )
        result = await greeting_agent_function(state)
        print("greeting_agent_function 결과:", result)

    asyncio.run(test_greeting_agent())