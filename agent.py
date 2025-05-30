import os
import sys
import operator
import asyncio
from typing import TypedDict, Annotated, Sequence, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage, ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI

# summary_module에서 함수 가져오기
from summary_module import summary_module_function
# mcp_tools에서 도구 함수 및 리스트 가져오기
from mcp_tools import mcp_kakaotalk_tool, mcp_sql_tool, tools

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# --- 1. Agent State 정의 ---
# AgentState는 거의 동일하게 유지되지만, messages는 이제 다양한 역할(user, assistant, tool)의 딕셔너리를 포함합니다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # 대화 기록용
    user_input: str                                  # 현재 사용자 입력
    user_id: str | None
    session_id: str | None
    retrieved_user_info: dict | None
    retrieved_status_info: dict | None
    summary: str | None
    error_message: str | None


# --- 4. 노드 실행 함수 정의 ---

# --- 메인 LLM 에이전트 함수 ---
async def main_agent_function(state: AgentState):
    print(f"\n--- 메인 LLM 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))

    if user_input:
        current_messages.append(HumanMessage(content=user_input)) # HumanMessage 객체로 추가
        print(f"메인 LLM 에이전트가 사용자 입력을 받았습니다: '{user_input}'")

    # Gemini LLM 사용:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
        # convert_system_message_to_human=True, # Gemini가 SystemMessage를 다르게 처리할 수 있으므로 호환성 확보
        # model_kwargs={"streaming": True} # 스트리밍을 model_kwargs로 전달
    )
    llm_with_tools = llm.bind_tools(tools)
    llm_response_message = await llm_with_tools.ainvoke(current_messages)

    print(f"메인 LLM 에이전트 응답 내용: {llm_response_message.content}")
    if llm_response_message.tool_calls:
        # tool_calls가 딕셔너리 리스트인 경우를 처리
        tool_call_info = []
        for tc in llm_response_message.tool_calls:
            if isinstance(tc, dict):
                tool_call_info.append((tc.get("name"), tc.get("args")))
            else:
                tool_call_info.append((tc.name, tc.args))
        print(f"메인 LLM 에이전트가 도구 호출을 요청합니다: {tool_call_info}")

    current_messages.append(llm_response_message) # AIMessage 객체를 메시지 기록에 추가

    return {
        "messages": current_messages,
        "user_input": ""
    }

def greeting_agent_function(state: AgentState):
    print(f"\n--- 인사 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY,
        max_retries=2
        # convert_system_message_to_human=True, # Gemini가 SystemMessage를 다르게 처리할 수 있으므로 호환성 확보
        # model_kwargs={"streaming": True} # 스트리밍을 model_kwargs로 전달
    )
    return {
        "messages": current_messages,
        "user_input": ""
    }

def filler_agent_function(state: AgentState):
    print(f"\n--- 채움 에이전트 실행 ---")
    user_input = state.get("user_input", "")
    current_messages = list(state.get("messages", []))

    return {
        "messages": current_messages,
        "user_input": ""
    }


# --- 5. 조건부 엣지 함수 정의 ---
def route_after_llm(state: AgentState) -> Literal["tool_executor", "__end__"]:
    print(f"\n--- 조건부 라우팅 (route_after_llm) 실행 ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"라우팅 결정: tool_executor (LLM이 {len(last_message.tool_calls)}개의 도구 호출 요청)")
        return "tool_executor"
    
    # 도구 호출이 없는 경우, LLM이 직접 답변한 것으로 간주하고 해당 턴의 그래프 실행을 종료합니다.
    # 제어권은 다시 외부의 while 루프로 돌아가 다음 사용자 입력을 기다립니다.
    print(f"라우팅 결정: __end__ (LLM 응답 완료, 다음 사용자 입력을 기다립니다)")
    return "__end__"


# --- 6. 그래프 빌드 ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("greeting_agent", greeting_agent_function)
workflow.add_node("filler_agent", filler_agent_function)
workflow.add_node("main_agent", main_agent_function)       # 메인 LLM 노드
tool_executor_node = ToolNode(tools)                     # ToolNode 정의
workflow.add_node("tool_executor", tool_executor_node)   # ToolNode를 그래프에 추가

# 엣지 설정
workflow.set_entry_point("main_agent") # 시작점을 agent_llm으로 변경

workflow.add_conditional_edges(
    "main_agent",  # agent_llm 노드 실행 후
    route_after_llm,  # 이 함수를 통해 다음 경로 결정
    {
        "tool_executor": "tool_executor", # route_after_llm이 "tool_executor" 반환 시
        # "Summary_Module": END,
        "__end__": END
    }
)

workflow.add_edge("tool_executor", "main_agent") # ToolNode 실행 후 다시 agent_llm으로 가서 결과 처리

app = workflow.compile()

# --- 8. 실행 (예제 호출) ---
if __name__ == "__main__":
    async def main():
        print("\n--- 대화형 AICC 그래프 실행 ---")
        session_id = f"session_{os.getpid()}"

        with open("prompts/main_agent_prompt.txt", "r") as file:
            main_agent_prompt = file.read()

        # 1. 대화 시작 전, 상태를 단 한번만 초기화합니다.
        conversation_state = {
            "messages": [SystemMessage(content=main_agent_prompt)],
            "user_input": "",
            "session_id": session_id,
            "user_id": "user_1234",
            "retrieved_user_info": None,
            "retrieved_status_info": None,
            "summary": None,
            "error_message": None
        }

        while True:
            # 2. 사용자 입력 받기
            user_input = input("\n사용자 입력 (종료하려면 엔터만 입력): ")
            if not user_input.strip():
                print("사용자가 대화를 종료했습니다.")
                break

            # 3. 현재 상태에 새로운 사용자 입력을 업데이트합니다.
            conversation_state["user_input"] = user_input
            
            # 최종 상태를 저장할 변수
            final_state = None

            # 4. astream_events v2를 사용하여 그래프를 단 한 번만 실행합니다.
            print("\n--- 스트리밍 및 단일 그래프 실행 시작 ---")
            async for event in app.astream_events(conversation_state, version="v2", config={"recursion_limit": 15}):
                kind = event["event"]
                # print(f"kind: {kind}", f"event: {event}")
                
                # 4-1. LLM 토큰 스트리밍을 실시간으로 출력합니다.
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                
                # 4-2. 그래프 실행이 '끝났을 때'의 이벤트를 확인합니다.
                # event["name"] == "main_agent"는 그래프의 진입점 노드 이름입니다.
                # 이 이름의 'end' 이벤트가 발생하면 해당 그래프의 실행이 모두 완료되었음을 의미합니다.
                if kind == "on_chain_end" and event["name"] == "LangGraph":
                    # 이벤트의 출력 데이터에 우리가 원하는 최종 상태가 담겨 있습니다.
                    final_state = event["data"]["output"]
            
            print("\n--- 스트리밍 및 단일 그래프 실행 종료 ---\n")

            # 5. 스트림에서 포착한 최종 상태로 다음 턴을 위해 대화 상태를 업데이트합니다.
            # ainvoke를 호출할 필요가 없어졌습니다.
            if final_state is not None:
                conversation_state = final_state
            else:
                # 예외 처리: 만약 어떤 이유로든 최종 상태를 얻지 못했다면 루프를 중단합니다.
                print("오류: 대화의 최종 상태를 얻지 못했습니다.")
                break
            
            # 6. '업데이트된 상태'의 마지막 메시지를 확인하여 while 루프를 탈출할지 결정합니다.
            last_message = conversation_state["messages"][-1]
            if isinstance(last_message, AIMessage) and ("endofsentence" in last_message.content):
                print("\nLLM이 대화를 종료했습니다.")
                break
        
        # 7. While 루프가 끝난 후 (대화가 완전히 종료된 후) 최종 요약을 수행합니다.
        print("\n--- 최종 대화 내용 요약 수행 ---")
        summary_result = await summary_module_function(conversation_state)
        final_summary = summary_result.get("summary")

        print("\n" + "="*50)
        print(f"[최종 요약]: {final_summary}")
        print("="*50)

    asyncio.run(main())