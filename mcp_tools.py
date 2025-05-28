from langchain_core.tools import tool

@tool
async def mcp_kakaotalk_tool(user_request: str):
    """
    MCP_KakaoTalk 더미 툴입니다.
    """
    print(f"--- MCP_KakaoTalk_Tool 실행 (ToolNode 경유) ---")
    print(f"MCP_KakaoTalk (Tool)이 다음 요청을 받았습니다: '{user_request}'")
    # --- KakaoTalk API 상호작용 플레이스홀더 ---
    tool_result = f"카카오톡 연동 작업이 성공적으로 처리되었습니다."
    print(f"MCP_KakaoTalk (Tool) 더미 툴 결과: {tool_result}")
    # --- KakaoTalk API 상호작용 플레이스홀더 종료 ---
    return tool_result # ToolNode는 도구의 직접적인 결과 문자열을 기대합니다.

@tool
async def mcp_sql_tool(query_details: str):
    """
    MCP_SQL 더미 툴입니다.
    """
    print(f"--- MCP_SQL_Tool 실행 (ToolNode 경유) ---")
    print(f"MCP_SQL (Tool)이 다음 상세 정보로 데이터베이스 쿼리를 시도합니다: '{query_details}'")
    # --- DB 상호작용 플레이스홀더 ---
    tool_result = f"SQL 데이터베이스 작업이 성공적으로 처리되었습니다."
    print(f"MCP_SQL (Tool) 더미 툴 결과: {tool_result}")
    # --- DB 상호작용 플레이스홀더 종료 ---
    return tool_result

# 사용될 도구 리스트
tools = [mcp_kakaotalk_tool, mcp_sql_tool] 