from __future__ import annotations

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class PlannerLocationAnalysis(BaseModel):
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


class MealRestaurantConditionsAnalysis(BaseModel):
    location: PlannerLocationAnalysis | None = None
    restaurant_names: list[str] = Field(default_factory=list)

    raw_food_terms: list[str] = Field(default_factory=list)
    normalized_categories: list[str] = Field(default_factory=list)
    menu_keywords: list[str] = Field(default_factory=list)
    preference_keywords: list[str] = Field(default_factory=list)

    excluded_food_terms: list[str] = Field(default_factory=list)
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

    parking_required: bool | None = None
    pet_allowed_required: bool | None = None
    foreign_menu_required: bool | None = None

    required_conditions: list[str] = Field(default_factory=list)
    preferred_conditions: list[str] = Field(default_factory=list)
    sort_by: Literal[
        "relevance",
        "distance",
        "price_low",
        "price_high",
        "name",
    ] = "relevance"
    limit: int = 3


class MealRequestAnalysis(BaseModel):
    request_id: str
    meal_type: Literal[
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "cafe",
        "other",
    ] = "other"
    preferred_time: str | None = None
    duration_minutes: int | None = None
    anchor_place_id: str | None = None
    restaurant_conditions: MealRestaurantConditionsAnalysis = Field(
        default_factory=MealRestaurantConditionsAnalysis
    )


class PlannerQuestionAnalysis(BaseModel):
    primary_intent: Literal[
        "create_plan",
        "modify_plan",
        "place_search",
        "route_question",
        "budget_question",
        "follow_up",
        "unknown",
    ] = "unknown"

    trip_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_minutes: int | None = None

    start_location: PlannerLocationAnalysis | None = None
    end_location: PlannerLocationAnalysis | None = None

    required_places: list[str] = Field(default_factory=list)
    preferred_places: list[str] = Field(default_factory=list)
    excluded_places: list[str] = Field(default_factory=list)
    fixed_order_places: list[str] = Field(default_factory=list)

    people: int | None = None
    total_trip_budget: int | None = None
    food_budget: int | None = None
    activity_budget: int | None = None
    transportation_budget: int | None = None
    budget_priority: Literal[
        "strict",
        "balanced",
        "experience_first",
        "unknown",
    ] = "unknown"

    meal_requests: list[MealRequestAnalysis] = Field(default_factory=list)

    transport_mode: Literal[
        "walk",
        "public_transport",
        "car",
        "mixed",
        "unknown",
    ] = "unknown"
    max_walking_distance_m: float | None = None
    max_walking_minutes: int | None = None
    route_priority: Literal[
        "shortest_distance",
        "shortest_time",
        "fewest_transfers",
        "preferred_order",
        "balanced",
        "unknown",
    ] = "unknown"

    preference_keywords: list[str] = Field(default_factory=list)
    indoor_preferred: bool | None = None
    weather: str | None = None

    modification_target: str | None = None
    modification_action: str | None = None

    unsupported_conditions: list[str] = Field(default_factory=list)
    condition_conflicts: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class GeneratedItineraryItem(BaseModel):
    item_id: str
    item_type: Literal[
        "place",
        "restaurant",
        "travel",
        "break",
    ]
    name: str
    start_time: str | None = None
    end_time: str | None = None
    duration_minutes: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    place_id: str | None = None
    restaurant_id: str | None = None
    estimated_cost: int | None = None
    notes: list[str] = Field(default_factory=list)


class PlannerGeneratedPlan(BaseModel):
    answer: str
    itinerary: list[GeneratedItineraryItem] = Field(default_factory=list)
    selected_place_names: list[str] = Field(default_factory=list)
    selected_restaurant_ids: list[str] = Field(default_factory=list)

    estimated_food_cost: int | None = None
    estimated_activity_cost: int | None = None
    estimated_transportation_cost: int | None = None
    estimated_total_cost: int | None = None

    budget_calculation_status: Literal[
        "complete",
        "partial",
        "cannot_determine",
    ] = "cannot_determine"
    budget_estimation_notes: list[str] = Field(default_factory=list)
    unverified_cost_items: list[str] = Field(default_factory=list)
    schedule_conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


PLANNER_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 서울 관광 일정 챗봇의 질문 분석기입니다.
사용자 질문을 PlannerQuestionAnalysis 구조로 변환하세요.

분석 규칙
1. 사용자가 말하지 않은 날짜, 시간, 위치, 예산, 인원을 만들지 마세요.
2. 관광지의 필수·선호·제외 조건과 방문 순서 고정 조건을 구분하세요.
3. 아침, 점심, 저녁, 간식, 카페 요청은 각각 MealRequestAnalysis로 분리하세요.
4. 각 식사 요청의 음식 종류, 메뉴, 가격, 위치, 주차, 반려동물 조건은
   restaurant_conditions에 기록하세요. 식사 위치는 관광지 위치와 분리합니다.
   예: "경복궁을 보고 종각에서 점심"은 required_places=["경복궁"]이고,
   점심 restaurant_conditions.location.text는 "종각"입니다.
5. 음식 표현은 raw_food_terms에 원문을 보존하고 임의의 음식 종류도 허용하세요.
6. "내 위치에서 출발"처럼 현재 위치가 필요하지만 좌표가 없으면 재질문하세요.
7. 기존 일정 수정 요청인데 이전 일정이 제공되지 않았다면 재질문하세요.
8. 총예산과 식비·관광비·교통비가 모순되면 condition_conflicts에 기록하세요.
9. 실시간 날씨 조회, 실시간 교통, 실제 대중교통 소요시간, 교통 혼잡,
   실시간 영업 여부, 예약, 대기시간은 현재 지원하지 않습니다.
10. 사용자가 직접 제공한 날씨는 weather에 보존할 수 있습니다.
11. 경로 저장, 회원별 일정 저장, 과거 일정 불러오기는 현재 프로젝트 범위에서
    제외되었으므로 unsupported_conditions에 기록하세요.
12. 관광 정보가 부족해도 사실을 만들지 말고 부분 일정으로 안내할 수 있도록
    조건을 그대로 보존하세요.
13. 사용 가능한 전체 시간이 명확하지 않으면 정확한 시각표를 강제로 만들지 마세요.
14. 사용자가 "하루 코스", "반나절", "오전 코스", "오후 코스"라고 요청하고
    방문할 장소나 식사 조건이 하나 이상 있다면 시작·종료 시각만을 이유로
    재질문하지 마세요. 정확한 시각은 비워 두고 방문 순서 중심으로 계획하세요.
    사용자가 말하지 않은 장소별 소요시간은 만들지 마세요.
15. "종각에서 점심", "경복궁 근처에서 저녁"처럼 식사 위치가 명시되면
    해당 위치는 반드시 만족해야 하는 restaurant_conditions의 필수 조건입니다.
""".strip(),
        ),
        (
            "human",
            """
[이전 일정 상태]
{previous_context}

[사용자 질문]
{question}
""".strip(),
        ),
    ]
)


PLANNER_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 제공된 서울 관광 문서와 음식점 검색 결과를 이용해 일정을 구성하는
AI 서울 여행 플래너입니다. PlannerGeneratedPlan 구조로 답하세요.

계획 작성 규칙
1. 제공된 관광 문서와 음식점 데이터에 없는 사실을 만들지 마세요.
2. 관광 문서에 장소명이나 운영 정보가 명시되지 않으면 확정 정보처럼 쓰지 마세요.
3. 실시간 날씨, 교통, 혼잡, 영업 상태, 예약, 대기시간을 추측하지 마세요.
4. 실제 경로 API 결과가 없으면 정확한 이동시간과 도보거리를 만들지 마세요.
5. 직선거리는 실제 도보거리나 소요시간으로 표현하지 마세요.
6. 운영시간이 자유 형식이거나 누락됐으면 방문 가능 여부를 단정하지 말고
   확인 필요 항목으로 표시하세요.
7. 사용자가 지정한 필수 장소와 식사 조건을 우선 적용하세요.
8. 일정 시간이 부족하면 억지로 모든 장소를 넣지 말고 schedule_conflicts에 기록하세요.
9. 비용이 확인되지 않은 항목을 0원으로 계산하지 말고 unverified_cost_items에 기록하세요.
10. 메뉴명에 인분 수가 없다면 해당 메뉴가 인원수에 충분하다고 단정하지 마세요.
11. 식당은 제공된 restaurant_id 중에서만 선택하세요.
12. 저장 기능은 제공하지 않습니다. 현재 응답에서 생성한 일정만 안내하세요.
13. 이전 일정이 제공된 수정 요청은 지정된 부분만 바꾸고 나머지는 최대한 유지하세요.
14. 정보가 부족하면 완성된 것처럼 꾸미지 말고 부분 일정과 추가 확인사항을 함께 제시하세요.
15. answer에는 사용자가 바로 읽을 수 있도록 방문 순서, 식사, 근거,
    미확인 사항을 한국어로 정리하세요.
16. itinerary에는 실제로 선택한 항목만 넣고, 모르는 좌표·비용·시간은 비워 두세요.
17. start_time, end_time, duration_minutes는 사용자 조건이나 제공 문서에 정확한 값이
    있을 때만 채우세요. "반나절"만으로 "90분 소요" 같은 값을 만들지 마세요.
18. 관광지 설명은 tourism_context의 문장으로 직접 뒷받침될 때만 작성하세요.
    문서가 없으면 장소명만 유지하고 운영시간·요금·소요시간 확인 필요로 표시하세요.
19. 식당의 위치 조건을 만족하지 못한 후보는 meal_context에 들어오지 않습니다.
    meal_context에 없는 식당을 추가하거나 더 가까운 곳이라고 추정하지 마세요.
20. 비용이 하나라도 확인되지 않았다면 unverified_cost_items에 반드시 기록하고,
    "미확인 비용 없음"이라고 쓰지 마세요.
""".strip(),
        ),
        (
            "human",
            """
[사용자 질문]
{question}

[구조화 조건]
{conditions}

[관광 문서 검색 결과]
{tourism_context}

[식사별 음식점 검색 결과]
{meal_context}

[기존 일정]
{previous_itinerary}

[검색 경고 및 제한]
{warnings}
""".strip(),
        ),
    ]
)


PLANNER_FALLBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
제공된 데이터만 사용해 서울 여행 일정을 한국어로 정리하세요.
정확한 거리, 이동시간, 비용, 영업 여부가 확인되지 않으면 추측하지 말고
확인 필요라고 표시하세요. 경로 저장 기능은 제공하지 않습니다.
""".strip(),
        ),
        (
            "human",
            """
질문: {question}
조건: {conditions}
관광 문서: {tourism_context}
식당 결과: {meal_context}
경고: {warnings}
""".strip(),
        ),
    ]
)
