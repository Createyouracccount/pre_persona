import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from LLM_module.utils.mcp_tools import tools
from LLM_module.utils.nodes import main_agent_function, greeting_agent_function, filler_agent_function
from LLM_module.utils.state import AgentState

load_dotenv()

# --- 조건부 엣지 함수 정의 ---
def route_after_llm(state: AgentState):
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
tool_executor_node = ToolNode(tools)
workflow.add_node("tool_executor", tool_executor_node)
workflow.set_entry_point("main_agent")
workflow.add_conditional_edges(
    "main_agent",
    route_after_llm,
    {
        "tool_executor": "tool_executor",
        "__end__": END
    }
)
workflow.add_edge("tool_executor", "main_agent")
app = workflow.compile()