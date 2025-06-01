import os
import asyncio
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from LLM_module.agent import app
from LLM_module.utils.nodes import summary_module_function
from LLM_module.utils.state import AgentState

def run():
    async def main():
        print("\n--- 대화형 AICC 그래프 실행 ---")
        session_id = f"session_{os.getpid()}"

        conversation_state: AgentState = {
            "messages": [],
            "user_input": "",
            "user_query": None,
            "session_id": session_id,
            "user_id": None,
            "retrieved_user_info": None,
            "retrieved_status_info": None,
            "summary": None,
            "error_message": None
        }

        while True:
            user_input = input("\n사용자 입력 (종료하려면 엔터만 입력): ")
            if not user_input.strip():
                print("사용자가 대화를 종료했습니다.")
                break
            conversation_state["user_input"] = user_input
            if user_input:
                conversation_state["messages"].append(HumanMessage(content=user_input))

            print(f"DEBUG main.py: messages BEFORE calling app: {conversation_state['messages']}")

            final_state = None
            print("\n--- 스트리밍 및 단일 그래프 실행 시작 ---")
            async for event in app.astream_events(conversation_state, version="v2", config={"recursion_limit": 15}):
                # if event.get('type') == "interrupt":
                #     print(event["payload"]["request"])
                #     user_input = input("입력: ")
                #     app.update_state(conversation_state, {"user_input": user_input}, as_node="human_node")
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                if kind == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event["data"]["output"]
            print("\n--- 스트리밍 및 단일 그래프 실행 종료 ---\n")

            if final_state is not None:
                print(f"DEBUG main.py: messages FROM final_state: {final_state['messages']}")
                conversation_state = final_state
            else:
                print("오류: 대화의 최종 상태를 얻지 못했습니다.")
                break
            last_message = conversation_state["messages"][-1]
            if isinstance(last_message, AIMessage) and ("endofsentence" in last_message.content):
                print("\nLLM이 대화를 종료했습니다.")
                break
        print("\n--- 최종 대화 내용 요약 수행 ---")
        # summary_result = await summary_module_function(conversation_state)
        # final_summary = summary_result.get("summary")
        # print("\n" + "="*50)
        # print(f"[최종 요약]: {final_summary}")
        print("="*50)
    asyncio.run(main())

if __name__ == "__main__":
    run()