import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import SingleServerMCPClient

# 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

# FastMCP용 클라이언트 생성 함수
async def create_client():
    return SingleServerMCPClient(
        url="http://localhost:8000/sse",  # FastMCP 서버 포트에 맞춰 설정
        transport="sse"
    )

# MCP Graph 생성 함수
def mcp_graph(client):
    tools = client.get_tools()
    print("🔧 MCP Tools:", tools)

    # LLM 설정 (OpenAI API 키 필요)
    api_key = "YOUR_OPENAI_API_KEY"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    # 챗봇 노드 정의
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 상태 그래프 정의
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder.compile(checkpointer=memory)

# 메인 함수
async def main():
    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": "1"},
        tags=["my-tag"]
    )
    async with await create_client() as client:
        agent = mcp_graph(client)
        response = await agent.ainvoke(
            {"messages": "서울 날씨 알려줘"},
            config=config
        )
        print("📨 Agent Response:", response)

# 실행
asyncio.run(main())
