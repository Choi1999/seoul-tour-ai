from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from llm_loader import get_llm
from rag import (
    evaluate_restaurant_budgets,
    search_restaurants_by_conditions,
)
from restaurant.restaurant_prompt import (
    RESTAURANT_ANALYSIS_PROMPT,
    RESTAURANT_ANSWER_PROMPT,
    RestaurantQuestionAnalysis,
)
from restaurant.restaurant_state import RestaurantConditions, RestaurantState


CURRENT_LOCATION_PHRASES = (
    "내 근처",
    "내 주변",
    "내 위치",
    "현재 위치",
    "현재위치",
    "지금 위치",
    "지금위치",
    "여기 근처",
    "여기 주변",
    "이 근처",
    "이 주변",
)

NAMED_LOCATION_PATTERN = re.compile(
    r"(?P<name>[가-힣A-Za-z0-9·._-]{1,24})(?:역)?\s*(?:근처|주변|인근|부근)"
)


def _asks_for_current_location(question: str) -> bool:
    compact = " ".join(str(question).split())
    return any(phrase in compact for phrase in CURRENT_LOCATION_PHRASES)


def _extract_named_location(question: str) -> str:
    """'종각 근처'처럼 명시된 기준 장소를 질문 원문에서 보정한다."""
    matches = list(NAMED_LOCATION_PATTERN.finditer(str(question)))
    if not matches:
        return ""

    name = matches[-1].group("name").strip()
    ignored = {
        "맛집",
        "식당",
        "음식점",
        "가게",
        "내",
        "여기",
        "이",
        "현재",
        "지금",
    }
    if name in ignored:
        return ""
    return name


def _normalize_location_condition(
    question: str,
    conditions: RestaurantConditions,
) -> RestaurantConditions:
    """LLM이 명시 장소를 current_location으로 오분류하는 경우를 보정한다."""
    normalized: RestaurantConditions = dict(conditions)
    location_value = normalized.get("location")
    location = dict(location_value or {})

    current_location_requested = _asks_for_current_location(question)
    extracted_location = _extract_named_location(question)
    location_text = str(location.get("text") or "").strip()
    location_type = str(location.get("location_type") or "unknown")

    if current_location_requested:
        location["location_type"] = "current_location"
        if not location_text or location_text in {
            "내 위치",
            "현재 위치",
            "지금 위치",
            "여기",
        }:
            location["text"] = "현재 위치"
        normalized["location"] = location
        return normalized

    if extracted_location:
        location["text"] = extracted_location
        if location_type in {"current_location", "unknown", ""}:
            location["location_type"] = "landmark"
        normalized["location"] = location
        return normalized

    if location_type == "current_location":
        if location_text and location_text not in {
            "내 위치",
            "현재 위치",
            "지금 위치",
            "여기",
        }:
            location["location_type"] = "landmark"
            normalized["location"] = location
        else:
            normalized.pop("location", None)

    return normalized


CONDITION_KEYS = {
    "primary_intent",
    "requested_fields",
    "restaurant_names",
    "referenced_restaurant_ids",
    "follow_up_reference",
    "location",
    "max_distance_m",
    "distance_basis",
    "food",
    "preference_keywords",
    "excluded_food_terms",
    "excluded_restaurant_keywords",
    "excluded_areas",
    "min_menu_price",
    "max_menu_price",
    "budget_amount",
    "budget_scope",
    "people",
    "price_query_type",
    "visit_day",
    "visit_time",
    "parking_required",
    "pet_allowed_required",
    "foreign_menu_required",
    "required_conditions",
    "preferred_conditions",
    "comparison_criteria",
    "sort_by",
    "limit",
}


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        key = text.casefold()

        if not text or key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _previous_context(state: RestaurantState) -> str:
    previous = {
        "previous_conditions": state.get("previous_conditions", {}),
        "previous_restaurants": state.get("previous_restaurants", []),
    }
    return json.dumps(previous, ensure_ascii=False, default=str)


def _analysis_to_state(
    analysis: RestaurantQuestionAnalysis,
    question: str,
) -> dict[str, Any]:
    payload = analysis.model_dump(exclude_none=True)
    conditions: RestaurantConditions = {
        key: value
        for key, value in payload.items()
        if key in CONDITION_KEYS
    }

    conditions = _normalize_location_condition(question, conditions)

    limit = conditions.get("limit", 5)
    conditions["limit"] = max(1, min(int(limit), 20))

    return {
        "conditions": conditions,
        "unsupported_conditions": payload.get("unsupported_conditions", []),
        "condition_conflicts": payload.get("condition_conflicts", []),
        "needs_clarification": payload.get("needs_clarification", False),
        "clarification_question": payload.get("clarification_question", ""),
    }


def analyze_query_node(state: RestaurantState) -> dict[str, Any]:
    question = str(state.get("question", "")).strip()

    if not question:
        return {
            "conditions": {"primary_intent": "unknown", "limit": 5},
            "needs_clarification": True,
            "clarification_question": "찾고 싶은 식당이나 메뉴 조건을 입력해주세요.",
            "search_status": "failed",
            "error": "질문이 비어 있습니다.",
        }

    try:
        structured_llm = get_llm().with_structured_output(
            RestaurantQuestionAnalysis,
            method="function_calling",
        )
        chain = RESTAURANT_ANALYSIS_PROMPT | structured_llm
        analysis = chain.invoke(
            {
                "question": question,
                "previous_context": _previous_context(state),
            }
        )
        return _analysis_to_state(analysis, question)

    except Exception as error:
        return {
            "conditions": {
                "primary_intent": "search",
                "food": {
                    "raw_food_terms": [],
                    "normalized_categories": [],
                    "menu_keywords": [],
                },
                "sort_by": "relevance",
                "limit": 5,
            },
            "semantic_query": question,
            "warnings": [
                "질문 구조화 분석에 실패하여 원문 질문으로 검색을 시도합니다."
            ],
            "search_status": "partial",
            "error": f"질문 분석 오류: {error}",
        }


def validate_conditions_node(state: RestaurantState) -> dict[str, Any]:
    conditions = dict(state.get("conditions", {}))
    conflicts = list(state.get("condition_conflicts", []))
    warnings_list = list(state.get("warnings", []))

    min_price = conditions.get("min_menu_price")
    max_price = conditions.get("max_menu_price")

    if (
        min_price is not None
        and max_price is not None
        and min_price > max_price
    ):
        conflicts.append("최소 메뉴 가격이 최대 메뉴 가격보다 큽니다.")

    people = conditions.get("people")
    if people is not None and people <= 0:
        conflicts.append("인원수는 1명 이상이어야 합니다.")

    budget_amount = conditions.get("budget_amount")
    if budget_amount is not None and budget_amount <= 0:
        conflicts.append("예산은 0원보다 커야 합니다.")

    max_distance = conditions.get("max_distance_m")
    if max_distance is not None and max_distance <= 0:
        conflicts.append("최대 거리는 0보다 커야 합니다.")

    location = conditions.get("location") or {}
    location_type = location.get("location_type")

    if location_type == "current_location" and not (
        location.get("latitude") is not None
        and location.get("longitude") is not None
    ):
        return {
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
            "needs_clarification": True,
            "clarification_question": (
                "현재 위치를 기준으로 검색하려면 위도·경도 또는 기준 장소를 "
                "알려주세요."
            ),
        }

    follow_up_reference = conditions.get("follow_up_reference") or {}
    if (
        conditions.get("primary_intent") == "follow_up"
        and follow_up_reference
        and not state.get("previous_restaurants")
        and not conditions.get("referenced_restaurant_ids")
    ):
        return {
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
            "needs_clarification": True,
            "clarification_question": (
                "이전 식당 목록을 확인할 수 없습니다. 식당명을 다시 알려주세요."
            ),
        }

    if conditions.get("visit_day") or conditions.get("visit_time"):
        warnings_list.append(
            "영업시간 데이터는 자유 형식이므로 현재 영업 여부를 자동 판정하지 않습니다."
        )

    if conflicts:
        return {
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
            "needs_clarification": True,
            "clarification_question": (
                "조건이 서로 충돌합니다. " + " ".join(_unique_strings(conflicts))
            ),
        }

    return {
        "conditions": conditions,
        "condition_conflicts": [],
        "warnings": _unique_strings(warnings_list),
    }


def _route_after_validation(state: RestaurantState) -> str:
    if state.get("needs_clarification") or state.get("condition_conflicts"):
        return "build_context"
    return "retrieve"


def retrieve_restaurants_node(state: RestaurantState) -> dict[str, Any]:
    question = str(state.get("question", "")).strip()
    conditions = state.get("conditions", {})

    try:
        result = search_restaurants_by_conditions(
            question,
            conditions,
            candidate_limit=max(
                int(conditions.get("limit", 5)) * 10,
                40,
            ),
            use_vector=True,
        )

        filtered = result.get("filtered_restaurants", [])
        status = str(result.get("search_status") or "no_results")
        if filtered and state.get("unsupported_conditions"):
            status = "partial"

        return {
            **result,
            "warnings": _unique_strings(
                [
                    *state.get("warnings", []),
                    *result.get("warnings", []),
                ]
            ),
            "search_status": status,
            "error": "",
        }

    except Exception as error:
        return {
            "candidate_restaurants": [],
            "candidate_restaurant_ids": [],
            "candidate_menus": [],
            "filtered_restaurants": [],
            "filtered_menus": [],
            "filter_failure_reasons": [],
            "search_status": "failed",
            "error": f"식당 검색 오류: {error}",
        }


def evaluate_budget_node(state: RestaurantState) -> dict[str, Any]:
    conditions = state.get("conditions", {})
    budget_amount = conditions.get("budget_amount")

    if budget_amount is None:
        return {"budget_evaluations": []}

    evaluations = evaluate_restaurant_budgets(
        state.get("filtered_restaurants", []),
        state.get("filtered_menus", []),
        budget_amount=int(budget_amount),
        people=int(conditions.get("people") or 1),
        budget_scope=str(conditions.get("budget_scope") or "group_total"),
    )

    return {"budget_evaluations": evaluations}


def build_context_node(state: RestaurantState) -> dict[str, Any]:
    restaurants = state.get("filtered_restaurants", [])
    menus = state.get("filtered_menus", [])
    menu_map: dict[str, list[dict[str, Any]]] = {}

    for menu in menus:
        restaurant_id = str(menu.get("restaurant_id", "")).strip()
        if restaurant_id:
            menu_map.setdefault(restaurant_id, []).append(menu)

    context_restaurants: list[dict[str, Any]] = []

    for restaurant in restaurants:
        restaurant_id = str(restaurant.get("restaurant_id", "")).strip()
        item = {
            "restaurant_id": restaurant_id,
            "name": restaurant.get("name"),
            "address": restaurant.get("address"),
            "representative_menu": restaurant.get("representative_menu"),
            "business_hours": restaurant.get("business_hours"),
            "rest_day": restaurant.get("rest_day"),
            "parking_available": restaurant.get("parking_available"),
            "pet_allowed": restaurant.get("pet_allowed"),
            "foreign_menu_available": restaurant.get("foreign_menu_available"),
            "introduction": restaurant.get("introduction"),
            "landmark": restaurant.get("landmark"),
            "distance_m": restaurant.get("distance_m"),
            "distance_basis": restaurant.get("distance_basis"),
            "menus": menu_map.get(restaurant_id, [])[:20],
        }
        context_restaurants.append(item)

    payload = {
        "conditions": state.get("conditions", {}),
        "restaurants": context_restaurants,
        "category_matches": state.get("category_matches", []),
        "budget_evaluations": state.get("budget_evaluations", []),
        "filter_failure_reasons": state.get("filter_failure_reasons", []),
        "suggested_relaxations": state.get("suggested_relaxations", []),
        "unsupported_conditions": state.get("unsupported_conditions", []),
        "condition_conflicts": state.get("condition_conflicts", []),
        "search_status": state.get("search_status", "no_results"),
    }

    places = [
        {
            "restaurant_id": restaurant.get("restaurant_id"),
            "name": restaurant.get("name"),
            "latitude": restaurant.get("latitude"),
            "longitude": restaurant.get("longitude"),
            "category": "restaurant",
        }
        for restaurant in restaurants
        if restaurant.get("latitude") is not None
        and restaurant.get("longitude") is not None
    ]

    return {
        "context": json.dumps(payload, ensure_ascii=False, default=str, indent=2),
        "places": places,
    }


def _fallback_answer(state: RestaurantState) -> str:
    if state.get("needs_clarification"):
        return state.get("clarification_question") or "검색 조건을 조금 더 알려주세요."

    if state.get("error") and not state.get("filtered_restaurants"):
        return (
            "식당 검색을 완료하지 못했습니다. "
            f"오류: {state.get('error')}"
        )

    restaurants = state.get("filtered_restaurants", [])

    if not restaurants:
        reasons = state.get("filter_failure_reasons", [])
        text = "조건에 맞는 식당을 찾지 못했습니다."
        if reasons:
            text += " " + " ".join(reasons)
        return text

    lines = ["검색 결과입니다."]

    for restaurant in restaurants:
        name = restaurant.get("name") or "식당명 미확인"
        address = restaurant.get("address") or "주소 미확인"
        representative = restaurant.get("representative_menu")
        lines.append(f"- {name}: {address}")
        if representative:
            lines.append(f"  대표 메뉴: {representative}")

    return "\n".join(lines)


def generate_answer_node(state: RestaurantState) -> dict[str, Any]:
    if state.get("needs_clarification"):
        return {"answer": _fallback_answer(state)}

    try:
        chain = RESTAURANT_ANSWER_PROMPT | get_llm() | StrOutputParser()
        answer = chain.invoke(
            {
                "question": state.get("question", ""),
                "context": state.get("context", "{}"),
                "warnings": json.dumps(
                    {
                        "warnings": state.get("warnings", []),
                        "unsupported_conditions": state.get(
                            "unsupported_conditions", []
                        ),
                        "error": state.get("error", ""),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
        return {"answer": answer}

    except Exception as error:
        warnings_list = _unique_strings(
            [
                *state.get("warnings", []),
                f"최종 답변 생성에 실패하여 기본 형식으로 출력했습니다: {error}",
            ]
        )
        return {
            "answer": _fallback_answer(state),
            "warnings": warnings_list,
        }


def create_restaurant_graph():
    workflow = StateGraph(RestaurantState)

    workflow.add_node("analyze", analyze_query_node)
    workflow.add_node("validate", validate_conditions_node)
    workflow.add_node("retrieve", retrieve_restaurants_node)
    workflow.add_node("evaluate_budget", evaluate_budget_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "retrieve": "retrieve",
            "build_context": "build_context",
        },
    )
    workflow.add_edge("retrieve", "evaluate_budget")
    workflow.add_edge("evaluate_budget", "build_context")
    workflow.add_edge("build_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


graph = create_restaurant_graph()
graph_app = graph


def run(
    question: str,
    previous_state: RestaurantState | None = None,
) -> dict[str, Any]:
    initial_state: RestaurantState = {
        "question": question,
        "warnings": [],
        "unsupported_conditions": [],
        "condition_conflicts": [],
        "filter_failure_reasons": [],
        "suggested_relaxations": [],
        "needs_clarification": False,
        "clarification_question": "",
        "search_status": "partial",
        "error": "",
    }

    if previous_state:
        initial_state["previous_conditions"] = previous_state.get(
            "conditions",
            previous_state.get("previous_conditions", {}),
        )
        initial_state["previous_restaurants"] = previous_state.get(
            "filtered_restaurants",
            previous_state.get("previous_restaurants", []),
        )

    return graph.invoke(initial_state)
