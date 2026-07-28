from typing import TypedDict, List, Dict, Any, Optional

class TravelGraphState(TypedDict):
    """
    서울 여행 챗봇 Graph State
    """
    question: str                                    # 사용자 질문
    intent: Optional[str]                            # 의도 분류 (CHATBOT 등)
    chatbot_result: Optional[Dict[str, Any]]         # 챗봇 검색 노드 반환 데이터
    final_output: Optional[Dict[str, Any]]           # summary 노드 최종 결과