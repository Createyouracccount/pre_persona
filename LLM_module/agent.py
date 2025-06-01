from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from LLM_module.utils.mcp_tools import tools
from LLM_module.utils.nodes import main_agent_function, greeting_agent_function, filler_agent_function#, human_node
from LLM_module.utils.state import AgentState
from LLM_module.utils.retriever_tool import retriever_node_function


# --- 조건부 엣지 함수 정의 ---
def start_conditional_function(state: AgentState):
    if state.get("user_query") is not None:
        return "main_agent"
    else:
        return "greeting_agent"

def greeting_conditional_function(state: AgentState):
    if state.get("user_query") is None:
        return "__end__"
    else:
        return "filler_agent"

def main_conditional_function(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tool_executor"
    return "__end__"

# --- 그래프 빌드 ---
workflow = StateGraph(AgentState)
workflow.add_node("greeting_agent", greeting_agent_function)
workflow.add_node("filler_agent", filler_agent_function)
workflow.add_node("main_agent", main_agent_function)
workflow.add_node("rag_tool", retriever_node_function)
tool_executor_node = ToolNode(tools)
workflow.add_node("tool_executor", tool_executor_node)
# workflow.add_node("human_node", human_node)

# workflow.set_entry_point("__start__")

workflow.add_conditional_edges(
    START,
    start_conditional_function,
    {
        "main_agent": "main_agent",
        "greeting_agent": "greeting_agent"
    }
)

workflow.add_conditional_edges(
    "greeting_agent",
    greeting_conditional_function,
    {
        "__end__": END,
        "filler_agent": "filler_agent"
    }
)

workflow.add_conditional_edges(
    "main_agent",
    main_conditional_function,
    {
        "tool_executor": "tool_executor",
        "__end__": END
    }
)

# workflow.add_edge("human_node", "greeting_agent")
workflow.add_edge("filler_agent", "rag_tool")
workflow.add_edge("rag_tool", "main_agent")
workflow.add_edge("tool_executor", "main_agent")
app = workflow.compile()