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
async def agent_llm_function(state: AgentState):
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


# --- 5. 조건부 엣지 함수 정의 ---
def route_after_llm(state: AgentState) -> Literal["tool_executor", "Summary_Module", "__end__"]:
    print(f"\n--- 조건부 라우팅 (route_after_llm) 실행 ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None # 마지막 메시지 가져오기

    if isinstance(last_message, AIMessage) and last_message.tool_calls: # AIMessage 인스턴스이고 tool_calls가 있는지 확인
        print(f"라우팅 결정: tool_executor (LLM이 {len(last_message.tool_calls)}개의 도구 호출 요청)")
        return "tool_executor"
    else:
        # LLM 응답에 '종료' 또는 'end'라는 키워드가 포함되어 있으면 바로 END로
        last_llm_content = last_message.content.lower() if hasattr(last_message, 'content') else ""
        if "종료" in last_llm_content or "end" in last_llm_content:
            print(f"라우팅 결정: __end__ (LLM 응답에 종료 키워드 포함)")
            return "__end__"
        print(f"라우팅 결정: Summary_Module (도구 호출 없음 또는 일반 대화 흐름)")
        return "Summary_Module"


# --- 6. 그래프 빌드 ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent_llm", agent_llm_function)       # 메인 LLM 노드
tool_executor_node = ToolNode(tools)                     # ToolNode 정의
workflow.add_node("tool_executor", tool_executor_node)   # ToolNode를 그래프에 추가

# 엣지 설정
workflow.set_entry_point("agent_llm") # 시작점을 agent_llm으로 변경

workflow.add_conditional_edges(
    "agent_llm",  # agent_llm 노드 실행 후
    route_after_llm,  # 이 함수를 통해 다음 경로 결정
    {
        "tool_executor": "tool_executor", # route_after_llm이 "tool_executor" 반환 시
        # "Summary_Module": END,
        "__end__": END
    }
)

workflow.add_edge("tool_executor", "agent_llm") # ToolNode 실행 후 다시 agent_llm으로 가서 결과 처리

app = workflow.compile()

# --- 7. 그래프 실행 함수 (이전과 유사) ---
async def run_aicc_graph_v2(user_input: str, session_id: str, user_id: str | None = None):
    initial_state = {
        "messages": [],
        "user_input": user_input,
        "session_id": session_id,
        "user_id": user_id,
        "retrieved_user_info": None,
        "retrieved_status_info": None,
        "summary": None,
        "error_message": None
    }
    print(f"\n--- AICC 그래프 V2 시작. 세션: {session_id}, 사용자 입력: '{user_input}' ---")

    # 단계별 로그를 위한 스트리밍 (디버깅용)
    print("\n--- 스트리밍 이벤트 시작 ---")
    async for event in app.astream_events(initial_state, version="v1", config={"recursion_limit": 15}):
        kind = event["event"]
        tags = event.get("tags", [])
        if kind == "on_chat_model_start":
            print(f"  EVENT: {kind} | Name: {event['name']} | Tags: {tags} | Input: {event['data'].get('input')}")
        elif kind == "on_chat_model_stream":
            chunk_content = event['data']['chunk'].content
            if chunk_content:
                print(f"    STREAM: {chunk_content}", end="", flush=True)
        elif kind == "on_chat_model_end":
            print(f"\n  EVENT: {kind} | Name: {event['name']} | Tags: {tags} | Output: {event['data'].get('output')}")
        elif kind == "on_tool_start":
            print(f"  EVENT: {kind} | Name: {event['name']} | Input: {event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"  EVENT: {kind} | Name: {event['name']} | Output: {event['data'].get('output')}")
    print("\n--- 스트리밍 이벤트 종료 ---\n")

    final_state = await app.ainvoke(initial_state, {"recursion_limit": 15}) # 재귀 제한 증가

    print(f"\n--- AICC 그래프 V2 실행 완료. 세션: {session_id} ---")
    print(f"최종 상태 메시지 기록:")
    for msg in final_state.get("messages", []):
        # BaseMessage 객체의 속성에 직접 접근
        role = getattr(msg, 'name', getattr(msg, 'type', 'unknown'))
        content = getattr(msg, 'content', '')
        tool_calls = getattr(msg, 'tool_calls', None)
        print(f"  [{role}]: {content}")
        if tool_calls:
            print(f"    Tool Calls: {tool_calls}")

    # --- 그래프 실행 후 별도로 요약 수행 ---
    summary_result = await summary_module_function(final_state)
    final_state["summary"] = summary_result["summary"]
    final_state["messages"] = summary_result["messages"]

    print(f"최종 요약: {final_state.get('summary')}")
    return final_state

# --- 8. 실행 (예제 호출) ---
if __name__ == "__main__":
    async def main():
        print("\n--- 대화형 AICC 그래프 실행 ---")
        session_id = f"session_{os.getpid()}"
        user_id = None
        while True:
            user_input = input("\n사용자 입력(엔터만 입력 시 종료): ")
            if not user_input.strip():
                print("종료합니다.")
                break
            result = await run_aicc_graph_v2(user_input, session_id, user_id=user_id)
            print(f"\n[최종 요약]: {result.get('summary')}")
            print("-"*40)
    asyncio.run(main())