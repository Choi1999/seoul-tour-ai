from __future__ import annotations

from typing import Any, Literal, TypedDict


class LocationCondition(TypedDict, total=False):
    """사용자 질문에서 추출한 위치 조건."""

    text: str
    # 사용자가 입력한 위치 원문
    # 예: "경복궁", "서울역", "강남역 근처"

    location_type: Literal[
        "landmark",
        "address",
        "area",
        "current_location",
        "restaurant",
        "unknown",
    ]
    # landmark: 관광지·랜드마크
    # address: 도로명·지번 주소
    # area: 구·동·역세권 등의 지역
    # current_location: 사용자의 현재 위치
    # restaurant: 특정 식당을 위치 기준으로 사용
    # unknown: 위치 종류를 판단하지 못한 상태

    latitude: float
    longitude: float
    # 질문에 좌표가 포함되었거나 위치 해석이 끝난 경우 사용


class FoodCondition(TypedDict, total=False):
    """사용자가 요청한 음식 종류와 메뉴 조건."""

    raw_food_terms: list[str]
    # 사용자가 실제로 사용한 음식 표현
    # 예: ["피자집"], ["얼큰한 국물 요리"]

    normalized_categories: list[str]
    # 검색 편의를 위해 정규화한 음식 종류
    # 예: "중국집" -> ["중식"]
    # 미리 정한 카테고리만 허용하지 않고 자유 문자열로 저장

    menu_keywords: list[str]
    # 실제 메뉴명으로 판단되는 단어
    # 예: ["페퍼로니 피자"], ["순대국밥"]


class BudgetEvaluation(TypedDict, total=False):
    """한 식당의 메뉴와 사용자 예산을 비교한 결과."""

    restaurant_id: str
    budget_amount: int
    people: int

    selected_menu_ids: list[str]
    # 예산 계산에 사용한 메뉴 ID

    selected_menus: list[dict[str, Any]]
    # 사용자 답변에 표시할 메뉴명, 가격, 수량 등의 조합

    estimated_total_price: int
    # 선택한 메뉴 조합의 예상 총액

    budget_status: Literal[
        "within_budget",
        "possibly_within_budget",
        "over_budget",
        "cannot_determine",
    ]
    # within_budget: 가격 기준으로 예산 이내임을 확인
    # possibly_within_budget: 인분 정보가 부족하여 가능성만 판단
    # over_budget: 계산한 메뉴 조합이 예산을 초과
    # cannot_determine: 가격 또는 인분 정보 부족으로 판단 불가

    serving_basis: Literal[
        "explicit",
        "inferred_from_menu_name",
        "unknown",
    ]
    # explicit: 메뉴명에 "2인분"처럼 인분 수가 명시됨
    # inferred_from_menu_name: "대", "라지", "세트" 등으로 추정
    # unknown: 인분 판단 근거가 없음

    confidence: Literal["high", "medium", "low"]
    notes: list[str]
    # 사용자에게 함께 보여줄 판단 근거와 주의사항


class CategoryMatch(TypedDict, total=False):
    """식당을 요청한 음식 종류로 판단한 근거."""

    restaurant_id: str
    requested_term: str
    # 사용자가 요청한 음식 표현
    # 예: "피자집", "중국집"

    matched_category: str
    # 정규화하거나 검색 결과에서 추론한 음식 종류

    matched_fields: list[str]
    # 일치가 확인된 원본 데이터 필드
    # 예: RSTR_NM, REPRSNT_MENU_NM, MENU_NM, RSTR_INTRCN_CONT

    matched_values: list[str]
    # 실제로 일치한 식당명·대표메뉴·메뉴명·소개 내용

    confidence: Literal["high", "medium", "low"]


class RestaurantConditions(TypedDict, total=False):
    """질문 분석 노드가 사용자 질문에서 추출한 식당 검색 조건."""

    primary_intent: Literal[
        "search",
        "detail",
        "menu",
        "compare",
        "budget",
        "follow_up",
        "unknown",
    ]
    # search: 식당 검색·추천
    # detail: 특정 식당의 주소·영업시간·편의정보 조회
    # menu: 메뉴와 가격 조회
    # compare: 여러 식당 비교
    # budget: 예산에 맞는 식당 또는 메뉴 조합 조회
    # follow_up: 이전 답변을 참조하는 후속 질문

    requested_fields: list[str]
    # 사용자가 확인하려는 정보
    # 예: address, hours, menu, parking, pet_allowed

    restaurant_names: list[str]
    # 질문에 직접 언급된 식당명

    referenced_restaurant_ids: list[str]
    # 이전 답변의 식당을 ID로 다시 참조할 때 사용

    follow_up_reference: dict[str, Any]
    # 예: "두 번째 식당", "거기", "그 집"과 같은 참조 정보

    location: LocationCondition

    max_distance_m: float
    # 기준 위치로부터 허용할 최대 거리

    distance_basis: Literal[
        "straight_line",
        "route",
        "landmark_distance",
        "unknown",
    ]
    # straight_line: 위도·경도로 계산한 직선거리
    # route: 외부 경로 API가 제공한 실제 이동거리
    # landmark_distance: 원본 데이터의 랜드마크 거리

    food: FoodCondition

    preference_keywords: list[str]
    # 정확한 값으로 필터링하기 어려운 의미 조건
    # 예: "든든한", "얼큰한", "가볍게 먹기 좋은"

    excluded_food_terms: list[str]
    excluded_restaurant_keywords: list[str]
    excluded_areas: list[str]
    # 사용자가 제외해 달라고 한 음식·식당 특징·지역

    min_menu_price: int
    max_menu_price: int

    budget_amount: int

    budget_scope: Literal[
        "single_menu",
        "per_person",
        "group_total",
        "unknown",
    ]
    # single_menu: 메뉴 하나의 가격 기준
    # per_person: 1인당 예산
    # group_total: 전체 인원의 총예산

    people: int

    price_query_type: Literal[
        "menu_price",
        "minimum_price",
        "maximum_price",
        "budget_combination",
        "unknown",
    ]

    visit_day: str
    visit_time: str
    # 영업시간·휴무일 확인에 사용할 방문 예정 요일과 시간

    parking_required: bool
    pet_allowed_required: bool
    foreign_menu_required: bool

    required_conditions: list[str]
    # 반드시 만족해야 하는 조건

    preferred_conditions: list[str]
    # 결과가 없을 때 사용자 동의를 받고 완화할 수 있는 선호 조건

    comparison_criteria: list[str]
    # 비교할 항목
    # 예: price, distance, menu, parking, business_hours

    sort_by: Literal[
        "relevance",
        "distance",
        "price_low",
        "price_high",
        "name",
    ]

    limit: int
    # 최종적으로 제시할 최대 식당 수


class RestaurantState(TypedDict, total=False):
    """Restaurant Graph의 노드 사이에서 공유하는 전체 실행 상태."""

    question: str
    # 사용자가 입력한 원문 질문

    conditions: RestaurantConditions
    # 질문 분석 노드가 추출한 구조화 조건

    semantic_query: str
    # 기존 ChromaDB에 전달할 의미 검색 문장
    # ChromaDB를 수정하거나 데이터를 추가하는 값이 아님

    expanded_search_terms: list[str]
    # 음식 종류의 동의어와 연관 검색어
    # 예: "피자집" -> "피자", "화덕피자", "피제리아"

    resolved_location: dict[str, Any]
    # 위치 해석 결과
    # 예: 장소명, 위도, 경도, 해석 성공 여부

    candidate_restaurants: list[dict[str, Any]]
    # ChromaDB 또는 키워드 검색에서 가져온 식당 후보

    candidate_restaurant_ids: list[str]
    # data_loader에서 상세정보와 메뉴를 조회할 식당 ID

    candidate_menus: list[dict[str, Any]]
    # 후보 식당에 연결된 전체 메뉴 데이터

    filtered_restaurants: list[dict[str, Any]]
    filtered_menus: list[dict[str, Any]]
    # 가격·거리·주차 등 정확 조건을 적용한 결과

    restaurant_distances: dict[str, float]
    # RSTR_ID를 키로 사용하는 식당별 거리

    category_matches: list[CategoryMatch]
    # 요청한 음식 종류와 후보 식당이 일치한 근거

    budget_evaluations: list[BudgetEvaluation]
    # 식당별 메뉴 조합과 예산 평가 결과

    comparison_result: dict[str, Any]
    # 식당 비교 기준과 비교 결과

    previous_conditions: RestaurantConditions
    previous_restaurants: list[dict[str, Any]]
    # "두 번째 식당은?", "거기는 주차돼?" 같은 후속 질문 지원용

    filter_failure_reasons: list[str]
    # 어떤 필수 조건 때문에 후보가 모두 제거됐는지 기록

    suggested_relaxations: list[str]
    # 자동 적용하지 않고 사용자에게 제안만 할 조건 완화 항목

    unsupported_conditions: list[str]
    # 현재 데이터로 확인할 수 없는 조건
    # 예: 실시간 대기시간, 예약 가능 여부, 실제 리뷰 평점

    condition_conflicts: list[str]
    # 서로 모순되거나 동시에 만족하기 어려운 조건

    context: str
    # 실제 검색·필터 결과를 최종 답변 LLM에 전달하기 위한 문맥

    answer: str
    # 사용자에게 보여줄 최종 자연어 답변

    places: list[dict[str, Any]]
    # Streamlit 카드와 지도에 표시할 최종 식당 목록

    warnings: list[str]
    # 데이터 누락, 거리 계산 불가 등 사용자에게 알릴 비치명적 문제

    search_status: Literal[
        "success",
        "no_results",
        "partial",
        "failed",
    ]
    # success: 조건에 맞는 결과를 정상적으로 찾음
    # no_results: 기술 오류는 없지만 조건에 맞는 식당이 없음
    # partial: 일부 조건만 확인하거나 적용함
    # failed: 기술 오류로 검색 실행에 실패함

    needs_clarification: bool

    clarification_question: str
    # 위치·식당명·후속 참조가 모호할 때 사용자에게 다시 물을 내용

    error: str
    # 예외 메시지 등 기술 오류만 저장
    # 결과 없음이나 지원 불가 조건은 error로 처리하지 않음