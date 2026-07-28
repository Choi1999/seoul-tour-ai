from typing import TypedDict, List, Dict, Any

class PlannerState(TypedDict):
    question: str                  # 사용자 질문
    context: str                   # RAG 검색 문맥
    answer: str                    # LLM 생성 답변
    places: List[Dict[str, Any]]   # 지도 표시용 장소 좌표 목록