from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.channels.last_value import LastValue
from langchain_openai import ChatOpenAI

# 1. 상태 스키마 정의
class AgentState(TypedDict):
    user_input: str
    response: Annotated[str, LastValue]
    # 분기 결정을 위한 키 추가
    next_node: Annotated[str, LastValue] # 또는 적절한 채널 타입

# 1. LLM 준비
llm = ChatOpenAI(model="gpt-4.1-nano")

# 2. 노드 함수 정의 (상태는 dict로 전달)
# branch_node는 이제 분기 결정을 딕셔너리 형태로 반환
def branch_node(state: AgentState) -> dict:
    user_input = state["user_input"]
    # 예시: "echo:"로 시작하면 {"next_node": "echo"}, 아니면 {"next_node": "llm"} 반환
    if user_input.startswith("echo:"):
        return {"next_node": "echo"}
    else:
        return {"next_node": "llm"}

def llm_node(state: AgentState) -> dict:
    user_input = state["user_input"]
    result = llm.invoke(user_input)
    return {"response": result.content}

def echo_node(state: AgentState) -> dict:
    user_input = state["user_input"]
    return {"response": f"Echo: {user_input[5:]}"}

# 3. 그래프 정의 (스키마 명시)
graph = StateGraph(AgentState)
graph.add_node("branch", branch_node)
graph.add_node("llm", llm_node)
graph.add_node("echo", echo_node)

# 분기 엣지를 add_conditional_edges로 추가
# 분기 결정 로직을 람다 함수로 분리
graph.add_conditional_edges(
    "branch",  # 분기 시작 노드
    lambda state: state["next_node"], # 상태 딕셔너리에서 'next_node' 키의 값을 추출하여 분기 결정
    {          # 'next_node' 값에 따른 다음 노드 매핑
        "llm": "llm",
        "echo": "echo",
    }
)

# 일반 엣지 추가 (분기된 노드에서 END로)
graph.add_edge("llm", END)
graph.add_edge("echo", END)

graph.set_entry_point("branch")

# 4. 실행 함수
def run_agent(user_input: str) -> str:
    agent = graph.compile()
    # 초기 상태 설정. next_node는 branch_node가 설정하므로 초기값 "" 또는 None 가능
    state = {"user_input": user_input, "response": "", "next_node": ""}
    result = agent.invoke(state)
    return result["response"]

if __name__ == "__main__":
    print(run_agent("echo:안녕1234"))
    print(run_agent("안녕! 반가워!"))
