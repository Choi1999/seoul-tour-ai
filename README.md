# seoul-tour-ai
## 담당자 작업 범위

맛집 추천 담당자는 다음 파일을 작성합니다.

- `restaurant_state.py`
- `restaurant_graph.py`
- `restaurant_prompts.py`

관광 계획 담당자는 다음 파일을 작성합니다.

- `planner_state.py`
- `planner_graph.py`
- `planner_prompts.py`

공통 파일과 `streamlit_app.py`는 임의로 수정하지 않습니다.

Streamlit 연결은 최종 통합 담당자가 진행합니다.

## Graph 실행 규칙

각 `_graph.py`는 다음 형식의 함수를 외부에 제공해야 합니다.

```python
def run(question: str) -> dict:
    result = graph.invoke({
        "question": question
    })

    return result

최종 결과에는 반드시 answer 키가 있어야 합니다.

{
    "answer": "최종 답변"
}

## 공통 함수 사용

Graph 파일에서 원본 엑셀을 직접 읽지 않습니다.

```python
from data_loader import (
    get_landmark_names,
    search_restaurants_by_landmark,
    search_restaurants_by_keyword,
    get_restaurants_by_ids,
    get_menus_by_restaurant_ids,
)

from rag import search_restaurants
from llm_loader import get_llm

##Graph 테스트

Graph 테스트는 Streamlit이 아니라 app.py로 진행합니다.

python app.py
from rag import search_restaurants
from llm_loader import get_llm
