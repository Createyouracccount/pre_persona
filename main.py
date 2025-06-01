import os
import asyncio
from langchain_core.messages import SystemMessage, AIMessage
from LLM_module.agent import app
from LLM_module.utils.nodes import summary_module_function

def run():
    async def main():
        print("\n--- 대화형 AICC 그래프 실행 ---")
        session_id = f"session_{os.getpid()}"

        with open("prompts/main_agent_prompt.txt", "r") as file:
            main_agent_prompt = file.read()

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
            user_input = input("\n사용자 입력 (종료하려면 엔터만 입력): ")
            if not user_input.strip():
                print("사용자가 대화를 종료했습니다.")
                break
            conversation_state["user_input"] = user_input
            final_state = None
            print("\n--- 스트리밍 및 단일 그래프 실행 시작 ---")
            async for event in app.astream_events(conversation_state, version="v2", config={"recursion_limit": 15}):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                if kind == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event["data"]["output"]
            print("\n--- 스트리밍 및 단일 그래프 실행 종료 ---\n")
            if final_state is not None:
                conversation_state = final_state
            else:
                print("오류: 대화의 최종 상태를 얻지 못했습니다.")
                break
            last_message = conversation_state["messages"][-1]
            if isinstance(last_message, AIMessage) and ("endofsentence" in last_message.content):
                print("\nLLM이 대화를 종료했습니다.")
                break
        print("\n--- 최종 대화 내용 요약 수행 ---")
        summary_result = await summary_module_function(conversation_state)
        final_summary = summary_result.get("summary")
        print("\n" + "="*50)
        print(f"[최종 요약]: {final_summary}")
        print("="*50)
    asyncio.run(main())

if __name__ == "__main__":
    run()