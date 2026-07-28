import math
import sys
from pathlib import Path
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from planner.planner_state import PlannerState
from planner.planner_prompt import PLANNER_PROMPT

# 현재 파일 기준 4단계 위 경로 추가 (llm_loader 호출용)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from llm_loader import init_custom_llm

# 경로 및 DB 로드
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
db = Chroma(embedding_function=embedding, persist_directory=str(DB_PATH))
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = init_custom_llm()


# --- 유틸리티 함수 (기존 작성 코드) ---
def format_docs(docs):
    result = ""
    for doc in docs:
        result += doc.page_content + "\n\n"
    return result

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # 지구 반지름
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


# --- LangGraph 워크플로우 구축 ---
def create_planner_graph():
    
    # 1. 문서 검색 및 검색 결과 내 장소 좌표 추출 노드
    def retrieve_node(state: PlannerState):
        question = state["question"]
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        # 검색된 문서 메타데이터에서 좌표 정보 추출
        places = []
        seen = set()
        
        for doc in docs:
            meta = doc.metadata or {}
            name = meta.get("name") or meta.get("title")
            lat = meta.get("lat") or meta.get("latitude")
            lon = meta.get("lon") or meta.get("longitude")
            
            if name and lat and lon and name not in seen:
                try:
                    places.append({
                        "name": str(name),
                        "lat": float(lat),
                        "lon": float(lon),
                        "category": meta.get("category", "관광지")
                    })
                    seen.add(name)
                except (ValueError, TypeError):
                    continue

        return {"context": context, "places": places}

    # 2. LLM 답변 생성 노드
    def generate_node(state: PlannerState):
        chain = PLANNER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"]
        })
        return {"answer": answer}

    # 3. 그래프 연결
    workflow = StateGraph(PlannerState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()