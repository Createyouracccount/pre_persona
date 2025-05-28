from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import asyncio

# --- 1. Agent State 정의 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add] # 대화 기록 저장용
    user_input: str                                  # 현재 사용자 입력
    next_node: str | None                            # 조건부 라우팅용
    user_id: str | None
    session_id: str | None
    retrieved_user_info: dict | None
    retrieved_status_info: dict | None
    summary: str | None
    error_message: str | None

# --- 2. 노드 실행 함수 정의 (플레이스홀더) ---

# --- Agent 함수 (LLM 상호작용) ---
async def agent_function(state: AgentState):
    print(f"--- 에이전트 실행: 현재 상태 ---")
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    current_messages = messages + [{"role": "user", "content": user_input}]

    print(f"에이전트가 사용자 입력을 받았습니다: {user_input}")
    # --- Gemini LLM 호출 플레이스홀더 ---
    llm_response_text = f"LLM 응답: '{user_input}' (from Agent)"
    print(f"에이전트 LLM 응답: {llm_response_text}")
    # --- Gemini LLM 호출 플레이스홀더 종료 ---

    next_node_decision = None
    if "카카오톡" in user_input.lower() or "카톡" in user_input.lower():
        next_node_decision = "MCP_KakaoTalk"
    elif "sql" in user_input.lower() or "데이터베이스" in user_input.lower() or "db" in user_input.lower():
        next_node_decision = "MCP_SQL"
    elif "전달" in user_input.lower():
        next_node_decision = "Agent_1"
    else:
        pass # conditional_function_1에서 결정하도록 함

    return {
        "messages": current_messages + [{"role": "assistant", "content": llm_response_text}],
        "user_input": "",
        "next_node": next_node_decision
    }

async def agent_1_function(state: AgentState):
    print(f"--- Agent_1 실행: 현재 상태 ---")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else "이전 메시지 없음"

    print(f"Agent_1이 다음을 기반으로 처리 중: '{last_message}'")
    llm_response_text = f"Agent_1의 LLM 응답 (기반: '{last_message}')"
    print(f"Agent_1 LLM 응답: {llm_response_text}")

    return {
        "messages": messages + [{"role": "assistant", "content": llm_response_text}],
    }

async def summary_module_function(state: AgentState):
    print(f"--- Summary_Module 실행: 현재 상태 ---")
    messages = state.get("messages", [])
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    print(f"Summary_Module이 대화를 요약 중입니다.")
    summary_text = f"대화 요약: ... ({len(messages)}개의 메시지 기반)"
    print(f"요약: {summary_text}")
    print(f"DB: 세션 {state.get('session_id')}에 대한 요약이 여기에 저장됩니다.")

    return {"summary": summary_text}

# --- 더미 툴 함수 (MCP - Mission Critical Platform) ---
async def mcp_kakaotalk_function(state: AgentState):
    """
    MCP_KakaoTalk 더미 툴입니다.
    실제 KakaoTalk API와 상호작용하는 대신, 고정된 한글 메시지를 반환합니다.
    """
    print(f"--- MCP_KakaoTalk 더미 툴 실행 ---")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    print(f"MCP_KakaoTalk이 다음 요청을 받았습니다 (더미): '{user_input if user_input else messages[-1]['content']}'")

    # --- KakaoTalk API 상호작용 플레이스홀더 ---
    tool_result = "카카오톡 연동 작업이 (더미로) 성공적으로 처리되었습니다."
    print(f"MCP_KakaoTalk 더미 툴 결과: {tool_result}")
    # --- KakaoTalk API 상호작용 플레이스홀더 종료 ---

    return {
        "messages": messages + [{"role": "tool", "name": "MCP_KakaoTalk", "content": tool_result}],
        "next_node": None
    }

async def mcp_sql_function(state: AgentState):
    """
    MCP_SQL 더미 툴입니다.
    실제 SQL 데이터베이스와 상호작용하는 대신, 고정된 한글 메시지를 반환합니다.
    - DB1: 유저 정보 (더미)
    - DB2: 환불/상태 정보 (더미)
    """
    print(f"--- MCP_SQL 더미 툴 실행 ---")
    messages = state.get("messages", [])
    query_details = "대화 기반의 SQL 쿼리 파라미터 (더미)"
    print(f"MCP_SQL이 다음 상세 정보로 데이터베이스 쿼리를 시도합니다 (더미): '{query_details}'")

    # --- DB 상호작용 플레이스홀더 ---
    tool_result = "SQL 데이터베이스 작업이 (더미로) 성공적으로 처리되었습니다. (예: 사용자 정보 조회 완료, 상태 업데이트 완료)"
    print(f"MCP_SQL 더미 툴 결과: {tool_result}")
    # --- DB 상호작용 플레이스홀더 종료 ---

    return {
        "messages": messages + [{"role": "tool", "name": "MCP_SQL", "content": tool_result}],
        "next_node": None
    }

# --- 3. 조건부 엣지 함수 정의 ---
def conditional_function_1(state: AgentState) -> Literal["MCP_KakaoTalk", "MCP_SQL", "Agent_1", "__end__"]:
    print(f"--- 조건부 함수 1 실행 ---")
    determined_next_node = state.get("next_node")
    messages = state.get("messages", [])
    last_llm_response = ""
    if messages and messages[-1]["role"] == "assistant":
        last_llm_response = messages[-1]["content"]

    print(f"조건부 함수 평가 중. Agent로부터 결정된 next_node: {determined_next_node}")
    print(f"마지막 LLM 응답: '{last_llm_response}'")

    if determined_next_node:
        if determined_next_node == "MCP_KakaoTalk":
            print("라우팅: MCP_KakaoTalk")
            return "MCP_KakaoTalk"
        elif determined_next_node == "MCP_SQL":
            print("라우팅: MCP_SQL")
            return "MCP_SQL"
        elif determined_next_node == "Agent_1":
            print("라우팅: Agent_1")
            return "Agent_1"

    if "고맙습니다" in last_llm_response or "안녕히 계세요" in last_llm_response:
        print("라우팅: Agent_1 (요약 또는 종료 가능성)")
        return "Agent_1"

    print("조건부 함수 1에서 기본값으로 Agent_1으로 라우팅합니다.")
    return "Agent_1"

# --- 4. 그래프 빌드 ---
workflow = StateGraph(AgentState)

workflow.add_node("Agent", agent_function)
workflow.add_node("MCP_KakaoTalk", mcp_kakaotalk_function) # 더미 툴로 교체됨
workflow.add_node("MCP_SQL", mcp_sql_function)           # 더미 툴로 교체됨
workflow.add_node("Summary_Module", summary_module_function)
workflow.add_node("Agent_1", agent_1_function)

workflow.add_edge(START, "Agent")

workflow.add_conditional_edges(
    "Agent",
    conditional_function_1,
    {
        "MCP_KakaoTalk": "MCP_KakaoTalk",
        "MCP_SQL": "MCP_SQL",
        "Agent_1": "Agent_1"
    }
)

workflow.add_edge("MCP_KakaoTalk", "Agent")
workflow.add_edge("MCP_SQL", "Agent")
workflow.add_edge("Agent_1", "Summary_Module")
workflow.add_edge("Summary_Module", END)

app = workflow.compile()

# --- 5. (선택 사항) DB 연결 헬퍼 ---
# from your_db_module import db_get_user_info, db_get_refund_status, db_save_summary

# --- 6. 그래프 실행 (예제 호출) ---
async def run_aicc_graph(user_input: str, session_id: str, user_id: str | None = None):
    initial_state = {
        "messages": [],
        "user_input": user_input,
        "session_id": session_id,
        "user_id": user_id,
        "next_node": None,
        "retrieved_user_info": None,
        "retrieved_status_info": None,
        "summary": None,
        "error_message": None
    }
    print(f"\n--- AICC 그래프 시작. 세션: {session_id}, 사용자 입력: '{user_input}' ---")
    # 스트리밍 대신 invoke 사용 (더미에서는 간단하게)
    final_state = await app.ainvoke(initial_state, {"recursion_limit": 10}) # 재귀 제한 추가

    # 이벤트 스트리밍을 사용하여 단계별 로그 확인
    # async for event in app.astream_events(initial_state, version="v1", {"recursion_limit": 10}):
    #     kind = event["event"]
    #     if kind == "on_chain_start":
    #         print(f"단계 시작: {event['name']} (입력: {event['data'].get('input')})")
    #     elif kind == "on_chain_end":
    #         print(f"단계 종료: {event['name']} (출력: {event['data'].get('output')})")

    print(f"\n--- AICC 그래프 실행 완료. 세션: {session_id} ---")
    print(f"최종 상태: {final_state}")
    return final_state

if __name__ == "__main__":

    async def main():
        session_1 = "dummy_session_kakaotalk_001"
        print(f"\n--- 시나리오 1: 카카오톡 관련 더미 툴 호출 ---")
        result1 = await run_aicc_graph("카카오톡 연동 문제 문의합니다.", session_1, user_id="user_dummy_1")
        print(f"세션 {session_1}의 요약: {result1.get('summary')}")

        print("\n---------------------------------------------------\n")

        session_2 = "dummy_session_sql_002"
        print(f"\n--- 시나리오 2: DB 관련 더미 툴 호출 ---")
        result2 = await run_aicc_graph("제 계정 정보를 데이터베이스에서 확인해주세요.", session_2, user_id="user_dummy_2")
        print(f"세션 {session_2}의 요약: {result2.get('summary')}")

        print("\n---------------------------------------------------\n")

        session_3 = "dummy_session_general_003"
        print(f"\n--- 시나리오 3: 일반적인 대화 후 요약 ---")
        result3 = await run_aicc_graph("상담 감사합니다. 좋은 하루 되세요.", session_3)
        print(f"세션 {session_3}의 요약: {result3.get('summary')}")

    asyncio.run(main())