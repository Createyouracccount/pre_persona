from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import asyncio
from langchain_core.tools import tool # 도구 정의를 위해 추가
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage, ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI # Gemini 사용을 위해 추가
from dotenv import load_dotenv
import os

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

# --- 3. Mock LLM 정의 (실제 LLM으로 교체 가능) ---
class MockLLMWithTools:
    def __init__(self, tools_list):
        self.tool_names = {t.name for t in tools_list}

    async def invoke(self, messages_history: Sequence[BaseMessage]): # 입력 타입도 BaseMessage로
        user_input = ""
        for msg in reversed(messages_history):
            if msg.type == "human": # HumanMessage의 경우 .type 속성으로 확인
                user_input = msg.content
                break
            if msg.type == "tool": # ToolMessage의 경우
                # tool_name = msg.name # ToolMessage에는 name 속성이 직접적으로 없을 수 있음. tool_call_id로 원본 호출과 연결
                tool_content = msg.content
                return AIMessage(content=f"LLM: 이전 도구의 결과 ('{tool_content}')를 확인했습니다. 다음으로 무엇을 도와드릴까요?")

        raw_tool_calls = []
        response_content = f"LLM 응답: '{user_input}'에 대해 생각 중입니다... (메인 LLM)"

        if "카카오톡" in user_input.lower() or "카톡" in user_input.lower():
            response_content = "LLM: 카카오톡 관련 작업을 시작하려고 합니다. 어떤 도움이 필요하신가요?"
            raw_tool_calls.append({
                "name": "mcp_kakaotalk_tool",
                "args": {"user_request": user_input},
                "id": f"call_kakao_{abs(hash(user_input))}_{len(messages_history)}"
            })
        elif "sql" in user_input.lower() or "데이터베이스" in user_input.lower() or "db" in user_input.lower():
            response_content = "LLM: SQL 데이터베이스 관련 작업을 시작하려고 합니다. 무엇을 조회하거나 변경하시겠습니까?"
            raw_tool_calls.append({
                "name": "mcp_sql_tool",
                "args": {"query_details": user_input},
                "id": f"call_sql_{abs(hash(user_input))}_{len(messages_history)}"
            })
        # ... (다른 조건들) ...
        elif "전달" in user_input.lower():
             response_content = "LLM: 네, 알겠습니다. 관련 내용을 전달하는 절차로 넘어가겠습니다."
        elif "고맙습니다" in user_input.lower() or "감사합니다" in user_input.lower() or "안녕히 계세요" in user_input.lower():
            response_content = "LLM: 천만에요. 도움이 되어 기쁩니다. 대화를 마무리하겠습니다."

        if raw_tool_calls:
            # raw_tool_calls는 리스트이므로 첫 번째 요소에 접근
            first_tool_call = raw_tool_calls[0]
            # ToolCall 객체를 직접 생성하는 대신 딕셔너리 형태로 전달
            tool_call_dict = {
                "name": first_tool_call["name"],
                "args": first_tool_call["args"],
                "id": first_tool_call["id"]
            }

            # 딕셔너리를 리스트에 담아서 AIMessage 생성
            return AIMessage(content=response_content, tool_calls=[tool_call_dict])
        else:
            return AIMessage(content=response_content)



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


# --- Agent_1 함수 (이전과 유사하게 유지) ---
async def agent_1_function(state: AgentState):
    print(f"\n--- Agent_1 실행 ---")
    messages = state.get("messages", [])
    # last_message_content = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else "이전 메시지 없음"
    last_message_content = messages[-1].content if messages and hasattr(messages[-1], 'content') else "이전 메시지 없음"

    print(f"Agent_1이 다음을 기반으로 처리 중: '{last_message_content}'")
    # Agent_1 자체의 LLM 호출 또는 로직 (플레이스홀더)
    agent_1_response_text = f"Agent_1의 응답 (기반: '{last_message_content}'). 이제 요약으로 넘어갑니다."
    print(f"Agent_1 응답: {agent_1_response_text}")

    return {
        "messages": messages + [AIMessage(content=agent_1_response_text, additional_kwargs={"name": "Agent_1"})],
    }

# --- 5. 조건부 엣지 함수 정의 ---
def route_after_llm(state: AgentState) -> Literal["tool_executor", "Agent_1", "__end__"]:
    print(f"\n--- 조건부 라우팅 (route_after_llm) 실행 ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None # 마지막 메시지 가져오기

    if isinstance(last_message, AIMessage) and last_message.tool_calls: # AIMessage 인스턴스이고 tool_calls가 있는지 확인
        print(f"라우팅 결정: tool_executor (LLM이 {len(last_message.tool_calls)}개의 도구 호출 요청)")
        return "tool_executor"
    else:
        # "고맙습니다", "전달" 등의 키워드가 LLM 응답에 포함되어 Agent_1로 가야 하는 경우,
        # 또는 단순히 도구 호출이 없는 경우 Agent_1로 진행
        last_llm_content = last_message.content.lower() if hasattr(last_message, 'content') else ""
        if "전달하는 절차로 넘어가겠습니다" in last_llm_content or \
           "대화를 마무리하겠습니다" in last_llm_content or \
           "천만에요" in last_llm_content: # LLM이 대화 마무리 또는 특정 흐름을 나타내는 경우
            print(f"라우팅 결정: Agent_1 (LLM 응답 내용 기반, 다음 단계로 진행 또는 요약 준비)")
            return "Agent_1"

        # 일반적인 LLM 응답 후 (도구 호출 없이) 또는 특정 키워드가 없다면 Agent_1로.
        # 이 부분은 에이전트의 상세 로직에 따라 __end__로 바로 가거나, 다른 노드로 갈 수 있습니다.
        # 현재 구조상 Agent_1이 요약 전 단계이므로 Agent_1로 보냅니다.
        print(f"라우팅 결정: Agent_1 (도구 호출 없음 또는 일반 대화 흐름)")
        return "Agent_1"


# --- 6. 그래프 빌드 ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent_llm", agent_llm_function)       # 메인 LLM 노드
tool_executor_node = ToolNode(tools)                     # ToolNode 정의
workflow.add_node("tool_executor", tool_executor_node)   # ToolNode를 그래프에 추가
workflow.add_node("Agent_1", agent_1_function)
workflow.add_node("Summary_Module", summary_module_function)

# 엣지 설정
workflow.set_entry_point("agent_llm") # 시작점을 agent_llm으로 변경

workflow.add_conditional_edges(
    "agent_llm",  # agent_llm 노드 실행 후
    route_after_llm,  # 이 함수를 통해 다음 경로 결정
    {
        "tool_executor": "tool_executor", # route_after_llm이 "tool_executor" 반환 시
        "Agent_1": "Agent_1",             # route_after_llm이 "Agent_1" 반환 시
        # "__end__": END                  # 필요시 END로 직접 가는 경로 추가 가능
    }
)

workflow.add_edge("tool_executor", "agent_llm") # ToolNode 실행 후 다시 agent_llm으로 가서 결과 처리
workflow.add_edge("Agent_1", "Summary_Module")
workflow.add_edge("Summary_Module", END)

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

    # 단계별 로그를 위한 스트리밍 (디버깅에 유용)
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

    print(f"최종 요약: {final_state.get('summary')}")
    return final_state

# --- 8. 실행 (예제 호출) ---
if __name__ == "__main__":
    async def main():
        session_1 = "v2_session_kakaotalk_001"
        print(f"\n--- 시나리오 1: 카카오톡 관련 더미 툴 호출 (V2) ---")
        result1 = await run_aicc_graph_v2("카카오톡 연동 어떻게 하나요?", session_1, user_id="user_v2_1")
        # print(f"세션 {session_1}의 요약: {result1.get('summary')}") # 요약은 최종 상태 출력에서 확인

        print("\n" + "="*50 + "\n")

        session_2 = "v2_session_sql_002"
        print(f"\n--- 시나리오 2: DB 관련 더미 툴 호출 (V2) ---")
        result2 = await run_aicc_graph_v2("내 데이터베이스 계정 정보 알려줘.", session_2, user_id="user_v2_2")
        # print(f"세션 {session_2}의 요약: {result2.get('summary')}")

        print("\n" + "="*50 + "\n")

        session_3 = "v2_session_general_003"
        print(f"\n--- 시나리오 3: 일반적인 대화 후 요약 (V2) ---")
        result3 = await run_aicc_graph_v2("오늘 날씨 좋네요. 상담 고맙습니다!", session_3)
        # print(f"세션 {session_3}의 요약: {result3.get('summary')}")

        print("\n" + "="*50 + "\n")
        session_4 = "v2_session_direct_summary_004"
        print(f"\n--- 시나리오 4: LLM이 바로 요약으로 가는 상황 시뮬레이션 (V2) ---")
        result4 = await run_aicc_graph_v2("네 전달해주세요.", session_4, user_id="user_v2_4")

    asyncio.run(main())