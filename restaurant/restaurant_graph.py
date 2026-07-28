import json
import requests
import pandas as pd
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

# 동일 폴더 모듈 임포트
from state import TravelGraphState
from prompt import SYSTEM_PROMPT


# =====================================================================
# 1. Helper 함수 (데이터 조회 및 API 연동)
# =====================================================================
def search_restaurant_db(query: str) -> List[Dict[str, Any]]:
    """
    [엑셀 / DB 데이터 조회]
    실제 프로젝트 시 pandas로 엑셀 파일(restaurants_selected 등)을 읽어 필터링합니다.
    """
    # 예시: 올려주신 엑셀 스키마(RSTR_NM, RSTR_RDNMADR 등) 기준
    mock_db_results = [
        {
            "RSTR_ID": 1088,
            "RSTR_NM": "펜앤커피",
            "RSTR_RDNMADR": "서울특별시 종로구 인사동5길 12",
            "RSTR_LA": 37.572857,
            "RSTR_LO": 126.985577,
            "REPRSNT_MENU_NM": "아메리카노",
            "menus": [
                {"MENU_NM": "아메리카노", "MENU_PRICE": "4,500원"},
                {"MENU_NM": "카페라떼", "MENU_PRICE": "5,000원"}
            ],
            "BSNS_TM_CN": "평일 07:30~20:00, 공휴일 11:00~19:00",
            "PRKG_POS_YN": "N",
            "PET_ENTRN_POSBL_YN": "N",
            "RSTR_INTRCN_CONT": "종각역 근처에 위치한 아늑한 카페입니다."
        },
        {
            "RSTR_ID": 1441,
            "RSTR_NM": "종로 닭한마리",
            "RSTR_RDNMADR": "서울특별시 종로구 종로5가 123",
            "RSTR_LA": 37.570000,
            "RSTR_LO": 127.000000,
            "REPRSNT_MENU_NM": "닭한마리(2인분)",
            "menus": [
                {"MENU_NM": "닭한마리(2인분)", "MENU_PRICE": "22,000원"},
                {"MENU_NM": "닭도리탕(大)", "MENU_PRICE": "46,000원"}
            ],
            "BSNS_TM_CN": "매일 11:00~22:00",
            "PRKG_POS_YN": "Y",
            "PET_ENTRN_POSBL_YN": "N",
            "RSTR_INTRCN_CONT": "국물이 일품인 종로의 대표 닭한마리 맛집입니다."
        }
    ]
    return mock_db_results


def fetch_external_api_info(lat: float, lng: float) -> Dict[str, Any]:
    """
    [외부 API 연동]
    식당의 위/경도를 받아 실시간 정보(카카오/네이버/공공데이터 API)를 가져옵니다.
    """
    # 실제 API 연동 예시:
    # url = f"https://api.example.com/status?lat={lat}&lng={lng}"
    # response = requests.get(url)
    # return response.json()
    
    # 가짜 API 응답 데이터 (현재 영업 여부, 실시간 거리 등)
    return {
        "is_open_now": True,
        "current_status": "영업 중"
    }


# =====================================================================
# 2. Chatbot 노드 (데이터 + API + Prompt 통합)
# =====================================================================
def chatbot_node(state: TravelGraphState) -> Dict[str, Any]:
    question = state.get("question", "")
    print(f"\n🔍 [chatbot_node] 질문 처리 및 검색 실행: '{question}'")

    # 1) DB/엑셀에서 식당 데이터 검색
    db_items = search_restaurant_db(question)

    # 2) 검색된 식당에 API 실시간 정보 결합
    enriched_items = []
    context_text = ""

    for item in db_items:
        # API 호출로 실시간 정보 획득
        api_data = fetch_external_api_info(item["RSTR_LA"], item["RSTR_LO"])
        item["realtime_status"] = api_data.get("current_status", " 정보 없음")
        enriched_items.append(item)

        # 3) LLM에 넘겨줄 {context} 문자열 만들기 (시스템 프롬프트 지침 규격)
        menu_str = "\n".join([f"  - {m['MENU_NM']} : {m['MENU_PRICE']}" for m in item["menus"]])
        
        context_text += f"""
식당명: {item['RSTR_NM']}
주소: {item['RSTR_RDNMADR']}
대표메뉴: {item['REPRSNT_MENU_NM']}
메뉴:
{menu_str}
영업시간: {item['BSNS_TM_CN']} (실시간 상태: {item['realtime_status']})
주차: {item['PRKG_POS_YN']}
반려동물: {item['PET_ENTRN_POSBL_YN']}
소개: {item['RSTR_INTRCN_CONT']}
----------------------------------------
"""

    # 4) prompt.py의 SYSTEM_PROMPT 내 {context} 채우기
    formatted_system_prompt = SYSTEM_PROMPT.format(context=context_text)

    # 5) 최종 결과 반환 (추후 LLM연동 시 formatted_system_prompt를 LLM에 주입)
    chatbot_response = {
        "status": "success",
        "formatted_prompt": formatted_system_prompt,  # LLM에 최종 주입될 완성된 프롬프트
        "query": question,
        "items": enriched_items
    }

    return {"chatbot_result": chatbot_response}


# =====================================================================
# 3. Summary 노드 및 LangGraph 워크플로우 구성
# =====================================================================
def summary_node(state: TravelGraphState) -> Dict[str, Any]:
    final_data = state.get("chatbot_result", {"status": "empty", "message": "결과가 없습니다."})
    return {"final_output": final_data}


workflow = StateGraph(TravelGraphState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("summary", summary_node)

workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", "summary")
workflow.add_edge("summary", END)

# 최종 컴파일된 그래프 객체
graph_app = workflow.compile()