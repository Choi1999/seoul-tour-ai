from __future__ import annotations

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class LocationAnalysis(BaseModel):
    text: str | None = None
    location_type: Literal[
        "landmark",
        "address",
        "area",
        "current_location",
        "restaurant",
        "unknown",
    ] = "unknown"
    latitude: float | None = None
    longitude: float | None = None


class FoodAnalysis(BaseModel):
    raw_food_terms: list[str] = Field(default_factory=list)
    normalized_categories: list[str] = Field(default_factory=list)
    menu_keywords: list[str] = Field(default_factory=list)


class RestaurantQuestionAnalysis(BaseModel):
    primary_intent: Literal[
        "search",
        "detail",
        "menu",
        "compare",
        "budget",
        "follow_up",
        "unknown",
    ] = "unknown"

    requested_fields: list[str] = Field(default_factory=list)
    restaurant_names: list[str] = Field(default_factory=list)
    referenced_restaurant_ids: list[str] = Field(default_factory=list)
    follow_up_reference: dict[str, Any] = Field(default_factory=dict)

    location: LocationAnalysis | None = None
    max_distance_m: float | None = None
    distance_basis: Literal[
        "straight_line",
        "route",
        "landmark_distance",
        "unknown",
    ] = "unknown"

    food: FoodAnalysis = Field(default_factory=FoodAnalysis)
    preference_keywords: list[str] = Field(default_factory=list)

    excluded_food_terms: list[str] = Field(default_factory=list)
    excluded_restaurant_keywords: list[str] = Field(default_factory=list)
    excluded_areas: list[str] = Field(default_factory=list)

    min_menu_price: int | None = None
    max_menu_price: int | None = None
    budget_amount: int | None = None
    budget_scope: Literal[
        "single_menu",
        "per_person",
        "group_total",
        "unknown",
    ] = "unknown"
    people: int | None = None
    price_query_type: Literal[
        "menu_price",
        "minimum_price",
        "maximum_price",
        "budget_combination",
        "unknown",
    ] = "unknown"

    visit_day: str | None = None
    visit_time: str | None = None
    parking_required: bool | None = None
    pet_allowed_required: bool | None = None
    foreign_menu_required: bool | None = None

    required_conditions: list[str] = Field(default_factory=list)
    preferred_conditions: list[str] = Field(default_factory=list)
    comparison_criteria: list[str] = Field(default_factory=list)

    sort_by: Literal[
        "relevance",
        "distance",
        "price_low",
        "price_high",
        "name",
    ] = "relevance"
    limit: int = 5

    unsupported_conditions: list[str] = Field(default_factory=list)
    condition_conflicts: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


RESTAURANT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 서울 음식점 챗봇의 질문 분석기입니다.
사용자 질문을 RestaurantQuestionAnalysis 구조로 변환하세요.

지원하는 데이터
- 식당명
- 도로명·지번 주소
- 위도·경도
- 식당 소개
- 대표 메뉴
- 전체 메뉴명과 가격
- 영업시간 원문
- 휴무일 원문
- 주차 가능 여부
- 반려동물 출입 가능 여부
- 외국어 메뉴 제공 여부
- 식당과 데이터에 등록된 주변 랜드마크 및 거리

분석 규칙
1. 사용자가 말하지 않은 위치, 가격, 인원, 시간, 편의 조건을 만들지 마세요.
2. 음식 표현은 raw_food_terms에 원문을 보존하세요.
3. 피자집, 쌀국수집, 타코집처럼 임의의 음식 종류도 그대로 보존하세요.
4. 특정 메뉴명이면 menu_keywords에도 기록하세요.
5. normalized_categories는 확실히 정규화할 수 있을 때만 기록하세요.
6. 위치 표현이 있으면 location.text에는 기준 장소명만 기록하세요.
   예: "종각 근처"는 "종각", "경복궁 주변"은 "경복궁"입니다.
   "근처", "주변"이 있는데 기준 위치가 없으면 needs_clarification을 true로 설정하세요.
7. current_location은 "내 근처", "내 주변", "현재 위치", "여기 주변"처럼
   사용자의 실제 현재 위치를 뜻하는 표현에만 사용하세요.
   "종각 근처", "경복궁 주변", "강남역 인근"처럼 장소명이 명시된 질문은
   절대로 current_location으로 분류하지 말고 landmark 또는 area로 분류하세요.
   실제 현재 위치를 요청했지만 좌표가 제공되지 않았다면 재질문하세요.
8. 서로 모순되는 가격·거리·제외 조건은 condition_conflicts에 기록하세요.
9. 필수 조건과 선호 조건을 구분하세요. 조건을 임의로 완화하지 마세요.
10. 실시간 영업 여부, 실시간 대기시간, 예약 가능 여부, 배달 여부,
    리뷰 평점, 실제 교통시간처럼 현재 데이터로 확인할 수 없는 항목은
    unsupported_conditions에 기록하세요.
11. 영업시간과 휴무일은 자유 형식 원문이므로 질문 분석 단계에서
    현재 영업 중이라고 판단하지 마세요.
12. 사용자가 단순히 메뉴 가격을 묻는 경우 detail보다 menu를 우선하세요.
13. 여러 식당을 비교하면 compare, 인원과 예산으로 메뉴 조합을 묻는다면 budget입니다.
14. "두 번째 식당", "거기"처럼 앞선 답변을 참조하면 follow_up입니다.
    이전 대화 정보가 제공되지 않았다면 재질문이 필요할 수 있습니다.
15. 결과 개수는 사용자가 지정하지 않으면 5개, 최대 20개로 분석하세요.
16. 경로 또는 추천 결과 저장 기능은 현재 프로젝트 범위에서 제외되었습니다.
    저장 요청은 unsupported_conditions에 기록하세요.
""".strip(),
        ),
        (
            "human",
            """
[이전 대화 상태]
{previous_context}

[사용자 질문]
{question}
""".strip(),
        ),
    ]
)


RESTAURANT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 서울 음식점 데이터 안내 챗봇입니다.
제공된 검색·필터 결과만 사용해 한국어로 답변하세요.

답변 규칙
1. 검색 결과에 없는 사실을 만들지 마세요.
2. 실시간 영업 여부, 대기시간, 예약, 배달, 리뷰 평점을 추측하지 마세요.
3. 영업시간과 휴무일은 원문을 보여주고 현재 영업 중이라고 단정하지 마세요.
4. 사용자의 필수 조건을 만족하지 않는 식당을 조건 충족 결과처럼 소개하지 마세요.
5. 결과가 없으면 어떤 조건에서 후보가 제외됐는지 설명하세요.
6. 조건 완화는 자동 적용하지 말고 선택지로만 제안하세요.
7. 메뉴 가격이 없거나 숫자로 확인되지 않으면 가격 미확인으로 표시하세요.
8. 메뉴명에 인분 수가 명시되지 않았다면 특정 인원이 충분히 먹을 수 있다고
   단정하지 마세요.
9. 메뉴명에 대, 라지, 세트가 있으면 여러 명이 먹을 가능성은 설명할 수 있지만
   추정임을 명확히 표시하세요.
10. 음식 종류 판단 근거가 있으면 대표 메뉴, 전체 메뉴, 소개 중 무엇이
    일치했는지 간단히 설명하세요.
11. 거리 정보는 context의 distance_m과 distance_basis가 있을 때만 표시하세요.
    원본 landmark_distance 값은 사용하지 마세요. 직선거리를 도보 거리나
    이동시간으로 바꾸지 마세요.
12. 지원하지 않는 조건은 무시하지 말고 확인할 수 없다고 알려주세요.
13. 결과 저장 및 이전 경로 불러오기 기능은 현재 제공하지 않습니다.
14. 식당별로 이름, 주소, 대표 메뉴, 관련 메뉴와 가격을 우선 제시하고,
    질문에 포함된 경우에만 영업시간·휴무일·주차·반려동물·외국어 메뉴를
    함께 정리하세요.
15. 식당이 많으면 핵심 결과만 간결하게 정리하세요.
16. context의 restaurants에 없는 식당을 추가하지 마세요.
17. "이 외에도 여러 식당이 있다"처럼 제공되지 않은 결과의 존재를 단정하지 마세요.
18. 위치가 검증되지 않은 후보는 context에 포함되지 않으므로 답변에도 언급하지 마세요.
""".strip(),
        ),
        (
            "human",
            """
[사용자 질문]
{question}

[검색 및 필터 결과]
{context}

[경고 및 제한]
{warnings}
""".strip(),
        ),
    ]
)
