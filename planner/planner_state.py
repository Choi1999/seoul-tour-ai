from __future__ import annotations

from typing import Any, Literal, TypedDict

from restaurant.restaurant_state import (
    BudgetEvaluation,
    LocationCondition,
    RestaurantConditions,
)


class MealRequest(TypedDict, total=False):
    """여행 일정에 포함할 한 번의 식사 요청."""

    request_id: str
    # 아침·점심·저녁 요청을 구분하는 내부 ID
    # 예: breakfast_1, lunch_1, dinner_1

    meal_type: Literal[
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "cafe",
        "other",
    ]

    preferred_time: str

    duration_minutes: int
    # 사용자가 원하는 식사 시간과 예상 체류시간

    anchor_place_id: str
    # 식사 전후 기준이 되는 관광지 또는 일정 항목 ID

    restaurant_conditions: RestaurantConditions
    # Restaurant Graph와 동일한 음식 종류·가격·주차 등의 조건을 재사용

    selected_restaurant_id: str
    # 일정 구성 과정에서 최종 선택한 식당 ID


class PlannerConditions(TypedDict, total=False):
    """질문 분석 노드가 사용자 질문에서 추출한 여행 일정 조건."""

    primary_intent: Literal[
        "create_plan",
        "modify_plan",
        "place_search",
        "route_question",
        "budget_question",
        "follow_up",
        "unknown",
    ]

    trip_date: str
    start_time: str
    end_time: str

    duration_minutes: int
    # 여행 날짜와 전체 사용 가능 시간

    start_location: LocationCondition
    end_location: LocationCondition
    # 출발지와 일정 종료 위치

    required_places: list[str]
    # 반드시 포함해야 하는 관광지

    preferred_places: list[str]
    # 가능하면 포함할 관광지

    excluded_places: list[str]
    # 일정에서 제외할 장소

    fixed_order_places: list[str]
    # 사용자가 방문 순서를 직접 지정한 장소

    people: int

    total_trip_budget: int
    food_budget: int
    activity_budget: int
    transportation_budget: int

    budget_priority: Literal[
        "strict",
        "balanced",
        "experience_first",
        "unknown",
    ]
    # strict: 총예산을 넘기지 않는 것이 최우선
    # balanced: 예산과 일정 품질을 함께 고려
    # experience_first: 일부 초과 가능성을 알리고 경험을 우선

    meal_requests: list[MealRequest]
    # 아침·점심·저녁마다 서로 다른 RestaurantConditions를 저장

    transport_mode: Literal[
        "walk",
        "public_transport",
        "car",
        "mixed",
        "unknown",
    ]

    max_walking_distance_m: float
    max_walking_minutes: int

    route_priority: Literal[
        "shortest_distance",
        "shortest_time",
        "fewest_transfers",
        "preferred_order",
        "balanced",
        "unknown",
    ]

    preference_keywords: list[str]
    # 여행 분위기와 장소 선호
    # 예: "전통적인", "사진 찍기 좋은", "조용한"

    indoor_preferred: bool

    weather: str
    # 사용자가 직접 제공한 날씨 또는 실내 선호 조건

    modification_target: str

    modification_action: str
    # 기존 일정 수정 요청
    # 예: target="점심", action="한식 대신 피자로 변경"


class RouteSegment(TypedDict, total=False):
    """일정의 두 장소 사이 이동 구간."""

    from_place_id: str
    to_place_id: str
    from_name: str
    to_name: str

    transport_mode: str
    distance_m: float
    duration_minutes: int

    distance_basis: Literal[
        "straight_line",
        "route_api",
        "dataset",
        "unknown",
    ]
    # 직선거리인지 실제 경로 API 거리인지 반드시 구분

    notes: list[str]


class ItineraryItem(TypedDict, total=False):
    """최종 일정표의 한 항목."""

    item_id: str

    item_type: Literal[
        "place",
        "restaurant",
        "travel",
        "break",
    ]

    name: str

    start_time: str
    end_time: str
    duration_minutes: int

    latitude: float
    longitude: float

    place_id: str
    restaurant_id: str

    estimated_cost: int

    notes: list[str]


class PlannerState(TypedDict, total=False):
    """Planner Graph의 노드 사이에서 공유하는 전체 실행 상태."""

    question: str
    # 사용자가 입력한 원문 질문

    conditions: PlannerConditions
    # 질문 분석 노드가 추출한 구조화 여행 조건

    semantic_query: str
    # 관광 문서를 검색하기 위해 기존 ChromaDB에 전달할 검색 문장

    candidate_places: list[dict[str, Any]]
    # RAG 검색에서 가져온 관광지 후보

    selected_places: list[dict[str, Any]]
    # 시간·위치·사용자 선호를 적용해 선택한 관광지

    meal_candidates: dict[str, list[dict[str, Any]]]
    # MealRequest.request_id별 식당 후보

    selected_restaurants: list[dict[str, Any]]
    # 각 식사 요청에 대해 최종 선택된 식당

    menus: list[dict[str, Any]]
    # 최종 선택 식당의 메뉴와 가격 정보

    route: list[RouteSegment]
    # 장소와 식당 사이 이동 구간

    itinerary: list[ItineraryItem]
    # 시간순으로 정렬된 최종 일정표

    total_distance_m: float

    route_distance_basis: Literal[
        "straight_line",
        "route_api",
        "mixed",
        "unknown",
    ]
    # 전체 일정에 사용한 거리 계산 기준

    estimated_food_cost: int
    estimated_activity_cost: int
    estimated_transportation_cost: int
    estimated_total_cost: int
    remaining_budget: int

    budget_exceeded: bool

    budget_calculation_status: Literal[
        "complete",
        "partial",
        "cannot_determine",
    ]
    # complete: 필요한 가격을 모두 확인함
    # partial: 일부 비용만 확인함
    # cannot_determine: 비용 계산 근거가 부족함

    budget_estimation_notes: list[str]

    unverified_cost_items: list[str]
    # 가격을 확인하지 못한 관광지·교통·메뉴 항목

    meal_budget_evaluations: dict[
        str,
        list[BudgetEvaluation],
    ]
    # MealRequest.request_id별 식사 예산 평가

    meal_failure_reasons: dict[str, list[str]]
    # 각 식사 조건에 맞는 식당을 찾지 못한 이유

    previous_itinerary: list[ItineraryItem]
    # "점심만 바꿔줘" 같은 후속 수정 요청에 사용할 기존 일정

    schedule_conflicts: list[str]
    # 운영시간·이동시간·종료시간 때문에 발생한 일정 충돌

    context: str
    # 실제 검색·계산 결과를 최종 답변 LLM에 전달하기 위한 문맥

    answer: str
    # 사용자에게 보여줄 최종 자연어 일정 답변

    places: list[dict[str, Any]]
    # Streamlit 지도에 표시할 관광지와 식당 목록

    warnings: list[str]
    # 거리·비용·운영시간 일부 미확인 등 비치명적 문제

    needs_clarification: bool

    clarification_question: str
    # 출발지·시간·필수 장소 등이 모호하여 재질문이 필요할 때 사용

    unsupported_conditions: list[str]

    condition_conflicts: list[str]
    # 지원 불가 조건과 서로 모순되는 조건

    search_status: Literal[
        "success",
        "no_results",
        "partial",
        "failed",
    ]

    error: str
    # 예외 메시지 등 기술 오류만 저장