from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from llm_loader import get_llm
from planner.planner_prompt import (
    PLANNER_ANALYSIS_PROMPT,
    PLANNER_FALLBACK_PROMPT,
    PLANNER_PLAN_PROMPT,
    PlannerGeneratedPlan,
    PlannerQuestionAnalysis,
)
from planner.planner_state import PlannerConditions, PlannerState
from rag import (
    evaluate_restaurant_budgets,
    search_restaurants_by_conditions,
    search_tourism_documents,
)


PLANNER_CONDITION_KEYS = {
    "primary_intent",
    "trip_date",
    "start_time",
    "end_time",
    "duration_minutes",
    "start_location",
    "end_location",
    "required_places",
    "preferred_places",
    "excluded_places",
    "fixed_order_places",
    "people",
    "total_trip_budget",
    "food_budget",
    "activity_budget",
    "transportation_budget",
    "budget_priority",
    "transport_mode",
    "max_walking_distance_m",
    "max_walking_minutes",
    "route_priority",
    "preference_keywords",
    "indoor_preferred",
    "weather",
    "modification_target",
    "modification_action",
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


def _load_previous_snapshot(state: PlannerState) -> dict[str, Any]:
    """run()이 전달한 이전 Planner 상태 요약을 읽는다."""
    raw_context = state.get("context")
    if not isinstance(raw_context, str) or not raw_context.strip():
        return {}

    try:
        payload = json.loads(raw_context)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    if not payload.get("__planner_previous_state__"):
        return {}

    return payload


def _previous_context(state: PlannerState) -> str:
    snapshot = _load_previous_snapshot(state)

    if snapshot:
        return json.dumps(
            {
                "previous_question": snapshot.get("previous_question", ""),
                "previous_conditions": snapshot.get("previous_conditions", {}),
                "previous_needs_clarification": snapshot.get(
                    "previous_needs_clarification", False
                ),
                "previous_clarification_question": snapshot.get(
                    "previous_clarification_question", ""
                ),
                "previous_itinerary": snapshot.get("previous_itinerary", []),
            },
            ensure_ascii=False,
            default=str,
        )

    return json.dumps(
        {
            "previous_itinerary": state.get("previous_itinerary", []),
        },
        ensure_ascii=False,
        default=str,
    )


def _is_meaningful_condition_value(value: Any) -> bool:
    """구조화 출력의 기본값이 이전의 유효한 조건을 덮지 않게 한다."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() != "unknown"
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _merge_condition_dicts(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """재질문 답변을 이전 일정 조건에 병합한다."""
    merged: dict[str, Any] = dict(previous)

    for key, value in current.items():
        if key == "meal_requests":
            if value:
                merged[key] = value
            elif key not in merged:
                merged[key] = []
            continue

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if _is_meaningful_condition_value(nested_value):
                    nested[nested_key] = nested_value
            merged[key] = nested
            continue

        if _is_meaningful_condition_value(value):
            merged[key] = value
        elif key not in merged:
            merged[key] = value

    return merged


def _is_broad_duration_request(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question)
    return any(
        keyword in normalized
        for keyword in (
            "반나절",
            "하루코스",
            "하루일정",
            "당일코스",
            "당일일정",
            "오전코스",
            "오후코스",
        )
    )


def _is_time_only_clarification(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False

    time_words = (
        "시작 시간",
        "종료 시간",
        "시작시간",
        "종료시간",
        "몇 시",
        "몇시",
        "시간대",
    )
    return any(word in text for word in time_words)


def _allow_broad_plan_without_exact_time(
    question: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """반나절·하루 요청은 정확한 시각이 없어도 순서 중심 일정으로 진행한다."""
    if not _is_broad_duration_request(question):
        return result

    conditions = dict(result.get("conditions") or {})
    has_plan_content = bool(
        conditions.get("required_places")
        or conditions.get("preferred_places")
        or conditions.get("meal_requests")
    )

    clarification_question = str(
        result.get("clarification_question") or ""
    )

    if (
        has_plan_content
        and result.get("needs_clarification")
        and _is_time_only_clarification(clarification_question)
    ):
        result["needs_clarification"] = False
        result["clarification_question"] = ""
        warnings_list = list(result.get("warnings") or [])
        warnings_list.append(
            "정확한 시작·종료 시간이 없어 방문 순서 중심의 일정으로 구성합니다."
        )
        result["warnings"] = _unique_strings(warnings_list)

    return result


MEAL_WORD_TO_TYPE = {
    "아침": "breakfast",
    "점심": "lunch",
    "저녁": "dinner",
    "간식": "snack",
    "카페": "cafe",
    "식사": "other",
}


def _repair_conditions_from_question(
    question: str,
    conditions: PlannerConditions,
) -> PlannerConditions:
    """명시적인 관광지·식사 위치 표현을 구조화 결과에 보강한다."""
    repaired: PlannerConditions = dict(conditions)
    required_places = list(repaired.get("required_places") or [])

    for match in re.finditer(
        r"([가-힣A-Za-z0-9·_-]+)(?:을|를)\s*"
        r"(?:보고|관람하고|둘러보고|방문하고|다녀오고)",
        question,
    ):
        place_name = match.group(1).strip()
        if place_name and place_name not in required_places:
            required_places.append(place_name)

    repaired["required_places"] = required_places
    meal_requests = [dict(item) for item in repaired.get("meal_requests") or []]

    location_matches = list(
        re.finditer(
            r"([가-힣A-Za-z0-9·_-]+)(?:\s*(?:근처|주변|인근))?에서\s*"
            r"(아침|점심|저녁|간식|카페|식사)",
            question,
        )
    )

    for location_match in location_matches:
        location_name = location_match.group(1).strip()
        meal_word = location_match.group(2)
        meal_type = MEAL_WORD_TO_TYPE[meal_word]

        target: dict[str, Any] | None = None
        for meal_request in meal_requests:
            if meal_request.get("meal_type") == meal_type:
                target = meal_request
                break

        if target is None:
            target = {
                "request_id": f"{meal_type}_{len(meal_requests) + 1}",
                "meal_type": meal_type,
                "preferred_time": "",
                "duration_minutes": None,
                "anchor_place_id": "",
                "restaurant_conditions": {
                    "primary_intent": "search",
                    "requested_fields": ["address", "menu", "price"],
                    "food": {
                        "raw_food_terms": [],
                        "normalized_categories": [],
                        "menu_keywords": [],
                    },
                    "preference_keywords": [],
                    "sort_by": "distance",
                    "limit": 3,
                },
                "selected_restaurant_id": "",
            }
            meal_requests.append(target)

        restaurant_conditions = dict(target.get("restaurant_conditions") or {})
        restaurant_conditions["location"] = {
            "text": location_name,
            "location_type": "landmark",
        }
        restaurant_conditions["sort_by"] = "distance"
        required_conditions = list(
            restaurant_conditions.get("required_conditions") or []
        )
        if "location" not in required_conditions:
            required_conditions.append("location")
        restaurant_conditions["required_conditions"] = required_conditions
        target["restaurant_conditions"] = restaurant_conditions

    repaired["meal_requests"] = meal_requests
    return repaired

def _convert_meal_request(meal: dict[str, Any]) -> dict[str, Any]:
    raw_conditions = dict(meal.get("restaurant_conditions") or {})

    food = {
        "raw_food_terms": raw_conditions.pop("raw_food_terms", []),
        "normalized_categories": raw_conditions.pop(
            "normalized_categories", []
        ),
        "menu_keywords": raw_conditions.pop("menu_keywords", []),
    }

    restaurant_conditions = {
        "primary_intent": "budget"
        if raw_conditions.get("budget_amount") is not None
        else "search",
        "requested_fields": ["address", "menu", "price"],
        "food": food,
        "preference_keywords": raw_conditions.pop(
            "preference_keywords", []
        ),
        "excluded_restaurant_keywords": [],
        "price_query_type": "budget_combination"
        if raw_conditions.get("budget_amount") is not None
        else "unknown",
        "distance_basis": "unknown",
        "visit_day": "",
        "visit_time": meal.get("preferred_time") or "",
        "comparison_criteria": [],
        **raw_conditions,
    }

    return {
        "request_id": meal.get("request_id"),
        "meal_type": meal.get("meal_type", "other"),
        "preferred_time": meal.get("preferred_time", ""),
        "duration_minutes": meal.get("duration_minutes"),
        "anchor_place_id": meal.get("anchor_place_id", ""),
        "restaurant_conditions": restaurant_conditions,
        "selected_restaurant_id": "",
    }


def _analysis_to_state(
    analysis: PlannerQuestionAnalysis,
    question: str,
    state: PlannerState,
) -> dict[str, Any]:
    payload = analysis.model_dump(exclude_none=True)
    conditions: PlannerConditions = {
        key: value
        for key, value in payload.items()
        if key in PLANNER_CONDITION_KEYS
    }
    conditions["meal_requests"] = [
        _convert_meal_request(meal)
        for meal in payload.get("meal_requests", [])
    ]

    snapshot = _load_previous_snapshot(state)
    previous_conditions = snapshot.get("previous_conditions")
    previous_was_clarification = bool(
        snapshot.get("previous_needs_clarification")
    )

    if isinstance(previous_conditions, dict) and (
        previous_was_clarification
        or payload.get("primary_intent") in {"follow_up", "modify_plan"}
    ):
        conditions = _merge_condition_dicts(
            previous_conditions,
            dict(conditions),
        )

    conditions = _repair_conditions_from_question(question, conditions)

    result = {
        "conditions": conditions,
        "unsupported_conditions": payload.get("unsupported_conditions", []),
        "condition_conflicts": payload.get("condition_conflicts", []),
        "needs_clarification": payload.get("needs_clarification", False),
        "clarification_question": payload.get("clarification_question", ""),
    }

    return _allow_broad_plan_without_exact_time(question, result)


def analyze_query_node(state: PlannerState) -> dict[str, Any]:
    question = str(state.get("question", "")).strip()

    if not question:
        return {
            "conditions": {"primary_intent": "unknown", "meal_requests": []},
            "needs_clarification": True,
            "clarification_question": "원하는 서울 여행 일정이나 장소를 입력해주세요.",
            "search_status": "failed",
            "error": "질문이 비어 있습니다.",
        }

    try:
        structured_llm = get_llm().with_structured_output(
            PlannerQuestionAnalysis,
            method="function_calling",
        )
        chain = PLANNER_ANALYSIS_PROMPT | structured_llm
        analysis = chain.invoke(
            {
                "question": question,
                "previous_context": _previous_context(state),
            }
        )
        return _analysis_to_state(analysis, question, state)

    except Exception as error:
        return {
            "conditions": {
                "primary_intent": "create_plan",
                "required_places": [],
                "preferred_places": [],
                "excluded_places": [],
                "fixed_order_places": [],
                "meal_requests": [],
                "preference_keywords": [],
                "transport_mode": "unknown",
                "route_priority": "unknown",
                "budget_priority": "unknown",
            },
            "semantic_query": question,
            "warnings": [
                "질문 구조화 분석에 실패하여 원문 질문으로 관광 문서를 검색합니다."
            ],
            "search_status": "partial",
            "error": f"질문 분석 오류: {error}",
        }


def validate_conditions_node(state: PlannerState) -> dict[str, Any]:
    conditions = dict(state.get("conditions", {}))
    conflicts = list(state.get("condition_conflicts", []))
    warnings_list = list(state.get("warnings", []))

    people = conditions.get("people")
    if people is not None and people <= 0:
        conflicts.append("인원수는 1명 이상이어야 합니다.")

    for key, label in (
        ("total_trip_budget", "총예산"),
        ("food_budget", "식비"),
        ("activity_budget", "관광비"),
        ("transportation_budget", "교통비"),
    ):
        value = conditions.get(key)
        if value is not None and value < 0:
            conflicts.append(f"{label}은 0원 이상이어야 합니다.")

    total_budget = conditions.get("total_trip_budget")
    category_budget_sum = sum(
        int(conditions.get(key) or 0)
        for key in (
            "food_budget",
            "activity_budget",
            "transportation_budget",
        )
    )

    if total_budget is not None and category_budget_sum > total_budget:
        conflicts.append(
            "식비·관광비·교통비의 합이 총예산보다 큽니다."
        )

    start_location = conditions.get("start_location") or {}
    if start_location.get("location_type") == "current_location" and not (
        start_location.get("latitude") is not None
        and start_location.get("longitude") is not None
    ):
        return {
            "needs_clarification": True,
            "clarification_question": (
                "현재 위치에서 출발하려면 위도·경도 또는 출발 장소를 알려주세요."
            ),
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
        }

    if (
        conditions.get("primary_intent") == "modify_plan"
        and not state.get("previous_itinerary")
    ):
        return {
            "needs_clarification": True,
            "clarification_question": (
                "수정할 이전 일정이 현재 대화 상태에 없습니다. "
                "기존 일정을 함께 입력해주세요."
            ),
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
        }

    if conflicts:
        return {
            "needs_clarification": True,
            "clarification_question": (
                "일정 조건이 서로 충돌합니다. " + " ".join(_unique_strings(conflicts))
            ),
            "condition_conflicts": _unique_strings(conflicts),
            "warnings": _unique_strings(warnings_list),
        }

    warnings_list.append(
        "현재 프로젝트는 실제 경로 API와 일정 저장 기능을 사용하지 않습니다. "
        "정확한 교통 소요시간과 저장된 경로는 제공하지 않습니다."
    )

    return {
        "conditions": conditions,
        "condition_conflicts": [],
        "warnings": _unique_strings(warnings_list),
    }


def _route_after_validation(state: PlannerState) -> str:
    if state.get("needs_clarification") or state.get("condition_conflicts"):
        return "build_context"
    return "retrieve_places"


def _build_tourism_query(question: str, conditions: PlannerConditions) -> str:
    terms = [
        *(conditions.get("required_places") or []),
        *(conditions.get("preferred_places") or []),
        *(conditions.get("preference_keywords") or []),
    ]
    text = " ".join(_unique_strings(terms))
    return text or question


def retrieve_places_node(state: PlannerState) -> dict[str, Any]:
    question = str(state.get("question", "")).strip()
    conditions = state.get("conditions", {})
    semantic_query = _build_tourism_query(question, conditions)
    required_places = _unique_strings(
        list(conditions.get("required_places") or [])
    )
    preferred_places = _unique_strings(
        list(conditions.get("preferred_places") or [])
    )

    queries = _unique_strings(
        [*required_places, *preferred_places, semantic_query]
    )
    documents: list[dict[str, Any]] = []
    warnings_list = list(state.get("warnings", []))
    vector_failed = False
    seen: set[tuple[Any, ...]] = set()

    for query in queries:
        try:
            result = search_tourism_documents(
                query,
                k=8 if query in required_places else 5,
                use_vector=True,
            )
        except Exception:
            result = {
                "documents": [],
                "warnings": [
                    "관광 문서 검색을 완료하지 못했습니다."
                ],
                "vector_failed": True,
            }

        vector_failed = vector_failed or bool(result.get("vector_failed"))
        warnings_list.extend(result.get("warnings", []))

        for document in result.get("documents", []):
            item = dict(document)
            matched_terms = list(item.get("matched_place_terms") or [])
            if query and query not in matched_terms:
                matched_terms.append(query)
            item["matched_place_terms"] = matched_terms
            key = (
                item.get("source"),
                item.get("page"),
                item.get("content"),
            )
            if key in seen:
                continue
            seen.add(key)
            documents.append(item)

    missing_required: list[str] = []
    for place_name in required_places:
        if not any(
            place_name.casefold()
            in str(document.get("content", "")).casefold()
            for document in documents
        ):
            missing_required.append(place_name)

    if missing_required:
        warnings_list.append(
            "관광 문서에서 직접 근거를 찾지 못한 필수 장소: "
            + ", ".join(missing_required)
        )

    return {
        "semantic_query": semantic_query,
        "candidate_places": documents[:20],
        "warnings": _unique_strings(warnings_list),
        "search_status": (
            "partial"
            if vector_failed or missing_required or not documents
            else "success"
        ),
    }

def retrieve_meals_node(state: PlannerState) -> dict[str, Any]:
    conditions = state.get("conditions", {})
    meal_requests = conditions.get("meal_requests", [])

    meal_candidates: dict[str, list[dict[str, Any]]] = {}
    meal_failure_reasons: dict[str, list[str]] = {}
    meal_budget_evaluations: dict[str, list[dict[str, Any]]] = {}
    selected_restaurants: list[dict[str, Any]] = []
    all_menus: list[dict[str, Any]] = []
    warnings_list = list(state.get("warnings", []))

    for index, meal_request in enumerate(meal_requests):
        request_id = str(
            meal_request.get("request_id") or f"meal_{index + 1}"
        )
        restaurant_conditions = dict(
            meal_request.get("restaurant_conditions") or {}
        )

        if restaurant_conditions.get("people") is None and conditions.get(
            "people"
        ) is not None:
            restaurant_conditions["people"] = conditions.get("people")

        meal_question = " ".join(
            part
            for part in (
                str(meal_request.get("meal_type") or "식사"),
                str(meal_request.get("preferred_time") or ""),
                str(state.get("question", "")),
            )
            if part.strip()
        )

        try:
            result = search_restaurants_by_conditions(
                meal_question,
                restaurant_conditions,
                candidate_limit=30,
                use_vector=True,
            )
        except Exception as error:
            meal_candidates[request_id] = []
            meal_failure_reasons[request_id] = [f"식당 검색 오류: {error}"]
            warnings_list.append(
                f"{request_id} 식당 검색에 실패했습니다: {error}"
            )
            continue

        filtered = result.get("filtered_restaurants", [])
        filtered_menus = result.get("filtered_menus", [])
        meal_candidates[request_id] = filtered
        all_menus.extend(filtered_menus)
        warnings_list.extend(result.get("warnings", []))

        if not filtered:
            meal_failure_reasons[request_id] = result.get(
                "filter_failure_reasons",
                ["조건에 맞는 식당을 찾지 못했습니다."],
            )
            continue

        selected = dict(filtered[0])
        selected["meal_request_id"] = request_id
        selected_restaurants.append(selected)
        meal_request["selected_restaurant_id"] = selected.get(
            "restaurant_id", ""
        )

        budget_amount = restaurant_conditions.get("budget_amount")
        if budget_amount is not None:
            meal_budget_evaluations[request_id] = evaluate_restaurant_budgets(
                filtered,
                filtered_menus,
                budget_amount=int(budget_amount),
                people=int(restaurant_conditions.get("people") or 1),
                budget_scope=str(
                    restaurant_conditions.get("budget_scope") or "group_total"
                ),
            )

    unique_menus: list[dict[str, Any]] = []
    seen_menu_keys: set[tuple[str, str]] = set()

    for menu in all_menus:
        key = (
            str(menu.get("restaurant_id", "")),
            str(menu.get("menu_id", "")),
        )
        if key in seen_menu_keys:
            continue
        seen_menu_keys.add(key)
        unique_menus.append(menu)

    return {
        "conditions": conditions,
        "meal_candidates": meal_candidates,
        "meal_failure_reasons": meal_failure_reasons,
        "meal_budget_evaluations": meal_budget_evaluations,
        "selected_restaurants": selected_restaurants,
        "menus": unique_menus,
        "warnings": _unique_strings(warnings_list),
    }


def build_context_node(state: PlannerState) -> dict[str, Any]:
    tourism_documents = [
        {
            "source": document.get("source"),
            "page": document.get("page"),
            "content": document.get("content"),
        }
        for document in state.get("candidate_places", [])[:12]
    ]

    meal_context: dict[str, Any] = {}
    menus = state.get("menus", [])

    for request_id, candidates in state.get("meal_candidates", {}).items():
        candidate_ids = {
            str(candidate.get("restaurant_id", ""))
            for candidate in candidates
        }
        meal_context[request_id] = {
            "restaurants": [
                {
                    "restaurant_id": candidate.get("restaurant_id"),
                    "name": candidate.get("name"),
                    "address": candidate.get("address"),
                    "representative_menu": candidate.get(
                        "representative_menu"
                    ),
                    "business_hours": candidate.get("business_hours"),
                    "rest_day": candidate.get("rest_day"),
                    "parking_available": candidate.get("parking_available"),
                    "pet_allowed": candidate.get("pet_allowed"),
                    "foreign_menu_available": candidate.get(
                        "foreign_menu_available"
                    ),
                    "latitude": candidate.get("latitude"),
                    "longitude": candidate.get("longitude"),
                }
                for candidate in candidates[:5]
            ],
            "menus": [
                menu
                for menu in menus
                if str(menu.get("restaurant_id", "")) in candidate_ids
            ][:30],
            "budget_evaluations": state.get(
                "meal_budget_evaluations", {}
            ).get(request_id, []),
            "failure_reasons": state.get("meal_failure_reasons", {}).get(
                request_id, []
            ),
        }

    context = {
        "tourism_documents": tourism_documents,
        "meal_context": meal_context,
        "unsupported_conditions": state.get("unsupported_conditions", []),
        "condition_conflicts": state.get("condition_conflicts", []),
        "warnings": state.get("warnings", []),
    }

    return {
        "context": json.dumps(context, ensure_ascii=False, default=str, indent=2)
    }


def _fallback_answer(state: PlannerState) -> str:
    if state.get("needs_clarification"):
        return state.get("clarification_question") or "일정 조건을 더 알려주세요."

    lines = [
        "확인 가능한 관광 문서와 식당 데이터만으로 부분 일정을 정리했습니다."
    ]

    if state.get("selected_restaurants"):
        lines.append("식사 후보")
        for restaurant in state.get("selected_restaurants", []):
            lines.append(
                f"- {restaurant.get('name') or '식당명 미확인'}: "
                f"{restaurant.get('address') or '주소 미확인'}"
            )

    if state.get("warnings"):
        lines.append("확인 필요")
        lines.extend(f"- {warning}" for warning in state.get("warnings", []))

    return "\n".join(lines)


def _document_supports_place(
    place_name: str,
    documents: list[dict[str, Any]],
) -> bool:
    place_name = str(place_name).strip()
    if not place_name:
        return False
    return any(
        place_name.casefold() in str(document.get("content", "")).casefold()
        for document in documents
    )


def _sanitize_itinerary(
    itinerary: list[dict[str, Any]],
    state: PlannerState,
) -> list[dict[str, Any]]:
    conditions = state.get("conditions", {})
    tourism_documents = list(state.get("candidate_places", []))
    requested_places = _unique_strings(
        [
            *(conditions.get("fixed_order_places") or []),
            *(conditions.get("required_places") or []),
            *(conditions.get("preferred_places") or []),
        ]
    )
    selected_restaurants = list(state.get("selected_restaurants", []))
    restaurant_by_id = {
        str(item.get("restaurant_id", "")): item
        for item in selected_restaurants
        if str(item.get("restaurant_id", ""))
    }
    restaurant_id_by_name = {
        str(item.get("name", "")).strip().casefold(): str(
            item.get("restaurant_id", "")
        )
        for item in selected_restaurants
        if str(item.get("name", "")).strip()
    }

    sanitized: list[dict[str, Any]] = []
    for raw_item in itinerary:
        item = dict(raw_item)
        item_type = str(item.get("item_type", ""))
        name = str(item.get("name", "")).strip()

        # 실제 경로 API가 없으므로 LLM이 만든 이동 구간은 저장하지 않는다.
        if item_type == "travel":
            continue

        if item_type == "restaurant":
            restaurant_id = str(item.get("restaurant_id", "")).strip()
            if not restaurant_id and name:
                restaurant_id = restaurant_id_by_name.get(name.casefold(), "")
            if restaurant_id not in restaurant_by_id:
                continue

            source = restaurant_by_id[restaurant_id]
            item["restaurant_id"] = restaurant_id
            item["name"] = source.get("name") or name
            item["latitude"] = source.get("latitude")
            item["longitude"] = source.get("longitude")
            item.pop("estimated_cost", None)

        elif item_type == "place":
            if requested_places:
                matched_requested = next(
                    (
                        place
                        for place in requested_places
                        if place.casefold() in name.casefold()
                        or name.casefold() in place.casefold()
                    ),
                    None,
                )
                if not matched_requested:
                    continue
                item["name"] = matched_requested
            elif not _document_supports_place(name, tourism_documents):
                continue

            item.pop("latitude", None)
            item.pop("longitude", None)
            item.pop("estimated_cost", None)

        elif item_type not in {"break"}:
            continue

        # 질문에 정확한 시각이 없으면 LLM이 임의로 시각표를 만들지 못하게 한다.
        if not conditions.get("start_time") and not conditions.get("end_time"):
            item.pop("start_time", None)
            item.pop("end_time", None)

        if item_type == "place":
            duration = item.get("duration_minutes")
            if duration is not None:
                duration_text = str(duration)
                supported = any(
                    _has_nearby_pattern(
                        str(document.get("content", "")),
                        str(item.get("name", "")),
                        rf"{re.escape(duration_text)}\s*분",
                    )
                    for document in tourism_documents
                )
                if not supported:
                    item.pop("duration_minutes", None)

        sanitized.append(item)

    return sanitized


def _has_nearby_pattern(
    text: str,
    anchor: str,
    pattern: str,
    *,
    window: int = 250,
) -> bool:
    text_folded = text.casefold()
    anchor_folded = anchor.casefold()
    start = 0

    while True:
        position = text_folded.find(anchor_folded, start)
        if position < 0:
            return False
        left = max(0, position - window)
        right = min(len(text), position + len(anchor) + window)
        if re.search(pattern, text[left:right], flags=re.IGNORECASE):
            return True
        start = position + len(anchor_folded)


def _build_unverified_cost_items(
    state: PlannerState,
    payload: dict[str, Any],
) -> list[str]:
    conditions = state.get("conditions", {})
    items = list(payload.get("unverified_cost_items", []))
    tourism_documents = list(state.get("candidate_places", []))

    for place_name in conditions.get("required_places", []) or []:
        supporting_text = " ".join(
            str(document.get("content", ""))
            for document in tourism_documents
            if place_name.casefold()
            in str(document.get("content", "")).casefold()
        )
        has_price_evidence = _has_nearby_pattern(
            supporting_text,
            place_name,
            r"(?:무료|\d[\d,]*\s*원)",
        )
        if not (
            has_price_evidence
            and payload.get("estimated_activity_cost") is not None
        ):
            items.append(f"{place_name} 입장료")

    stop_count = len(conditions.get("required_places", []) or []) + len(
        state.get("selected_restaurants", [])
    )
    if stop_count >= 2:
        # 실제 경로 API가 없으므로 LLM이 교통비 숫자를 만들더라도 확정하지 않는다.
        items.append("장소 간 교통비")

    for restaurant in state.get("selected_restaurants", []):
        name = restaurant.get("name") or "선택 식당"
        items.append(f"{name} 실제 주문 금액")

    return _unique_strings(items)

def _strip_unsupported_duration_claims(
    answer: str,
    state: PlannerState,
) -> str:
    source_text = " ".join(
        str(document.get("content", ""))
        for document in state.get("candidate_places", [])
    )
    source_text += " " + str(state.get("question", ""))

    pattern = re.compile(
        r"(?:약\s*)?(\d{1,3})\s*분"
        r"(?:\s*소요(?:\s*예상)?)?(?:입니다|이다)?"
    )

    def replace(match: re.Match[str]) -> str:
        number = match.group(1)
        if re.search(rf"{re.escape(number)}\s*분", source_text):
            return match.group(0)
        return "소요시간은 확인이 필요합니다"

    return pattern.sub(replace, answer)


def _grounded_partial_answer(state: PlannerState) -> str:
    conditions = state.get("conditions", {})
    lines = ["확인 가능한 데이터만으로 방문 순서를 정리했습니다."]

    place_order = _unique_strings(
        [
            *(conditions.get("fixed_order_places") or []),
            *(conditions.get("required_places") or []),
            *(conditions.get("preferred_places") or []),
        ]
    )
    for index, place_name in enumerate(place_order, start=1):
        if _document_supports_place(place_name, state.get("candidate_places", [])):
            lines.append(f"{index}. {place_name} 방문")
            lines.append("   - 관련 관광 문서를 확인했습니다.")
        else:
            lines.append(f"{index}. {place_name} 방문 요청")
            lines.append("   - 운영시간·요금·권장 소요시간은 확인이 필요합니다.")

    offset = len(place_order)
    for index, restaurant in enumerate(
        state.get("selected_restaurants", []),
        start=offset + 1,
    ):
        lines.append(f"{index}. 식사: {restaurant.get('name') or '식당명 미확인'}")
        lines.append(f"   - 주소: {restaurant.get('address') or '주소 미확인'}")
        if restaurant.get("business_hours"):
            lines.append(f"   - 영업시간 원문: {restaurant.get('business_hours')}")
        if restaurant.get("rest_day"):
            lines.append(f"   - 휴무일 원문: {restaurant.get('rest_day')}")

    lines.append(
        "정확한 이동시간과 실제 경로는 경로 API를 사용하지 않아 제공하지 않습니다."
    )
    return "\n".join(lines)


def generate_plan_node(state: PlannerState) -> dict[str, Any]:
    if state.get("needs_clarification"):
        return {
            "answer": _fallback_answer(state),
            "itinerary": [],
            "places": [],
            "search_status": "partial",
        }

    conditions = state.get("conditions", {})
    tourism_documents = list(state.get("candidate_places", []))
    required_places = list(conditions.get("required_places") or [])

    # 필수 관광지에 대한 문서가 하나도 없으면 일반지식으로 채우지 않는다.
    if required_places and not any(
        _document_supports_place(place, tourism_documents)
        for place in required_places
    ):
        unverified_items = _build_unverified_cost_items(state, {})
        return {
            "answer": _grounded_partial_answer(state),
            "itinerary": [],
            "selected_places": [],
            "places": [
                {
                    "restaurant_id": restaurant.get("restaurant_id"),
                    "name": restaurant.get("name"),
                    "latitude": restaurant.get("latitude"),
                    "longitude": restaurant.get("longitude"),
                    "category": "restaurant",
                }
                for restaurant in state.get("selected_restaurants", [])
                if restaurant.get("latitude") is not None
                and restaurant.get("longitude") is not None
            ],
            "unverified_cost_items": unverified_items,
            "budget_calculation_status": "cannot_determine",
            "schedule_conflicts": [],
            "route": [],
            "route_distance_basis": "unknown",
            "warnings": state.get("warnings", []),
            "search_status": "partial",
        }

    tourism_context = json.dumps(
        tourism_documents[:20],
        ensure_ascii=False,
        default=str,
    )
    meal_context = json.dumps(
        {
            "meal_candidates": state.get("meal_candidates", {}),
            "selected_restaurants": state.get("selected_restaurants", []),
            "menus": state.get("menus", []),
            "meal_budget_evaluations": state.get(
                "meal_budget_evaluations", {}
            ),
            "meal_failure_reasons": state.get("meal_failure_reasons", {}),
        },
        ensure_ascii=False,
        default=str,
    )

    try:
        structured_llm = get_llm().with_structured_output(
            PlannerGeneratedPlan,
            method="function_calling",
        )
        chain = PLANNER_PLAN_PROMPT | structured_llm
        plan = chain.invoke(
            {
                "question": state.get("question", ""),
                "conditions": json.dumps(
                    conditions,
                    ensure_ascii=False,
                    default=str,
                ),
                "tourism_context": tourism_context,
                "meal_context": meal_context,
                "previous_itinerary": json.dumps(
                    state.get("previous_itinerary", []),
                    ensure_ascii=False,
                    default=str,
                ),
                "warnings": json.dumps(
                    {
                        "warnings": state.get("warnings", []),
                        "unsupported_conditions": state.get(
                            "unsupported_conditions", []
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
        payload = plan.model_dump(exclude_none=True)

    except Exception as structured_error:
        try:
            fallback_chain = (
                PLANNER_FALLBACK_PROMPT | get_llm() | StrOutputParser()
            )
            answer = fallback_chain.invoke(
                {
                    "question": state.get("question", ""),
                    "conditions": json.dumps(
                        conditions,
                        ensure_ascii=False,
                        default=str,
                    ),
                    "tourism_context": tourism_context,
                    "meal_context": meal_context,
                    "warnings": json.dumps(
                        state.get("warnings", []),
                        ensure_ascii=False,
                    ),
                }
            )
            payload = {
                "answer": answer,
                "itinerary": [],
                "selected_place_names": [],
                "selected_restaurant_ids": [],
                "budget_calculation_status": "cannot_determine",
                "warnings": [
                    "구조화 일정 생성에 실패하여 텍스트 일정으로 출력했습니다: "
                    f"{structured_error}"
                ],
            }
        except Exception as fallback_error:
            payload = {
                "answer": _grounded_partial_answer(state),
                "itinerary": [],
                "selected_place_names": [],
                "selected_restaurant_ids": [],
                "budget_calculation_status": "cannot_determine",
                "warnings": [
                    "최종 일정 생성에 실패하여 기본 형식으로 출력했습니다: "
                    f"{fallback_error}"
                ],
            }

    itinerary = _sanitize_itinerary(
        list(payload.get("itinerary", [])),
        state,
    )

    selected_restaurant_ids = {
        str(value)
        for value in payload.get("selected_restaurant_ids", [])
    }
    allowed_restaurant_ids = {
        str(restaurant.get("restaurant_id", ""))
        for restaurant in state.get("selected_restaurants", [])
    }
    selected_restaurant_ids &= allowed_restaurant_ids
    selected_restaurants = [
        restaurant
        for restaurant in state.get("selected_restaurants", [])
        if not selected_restaurant_ids
        or str(restaurant.get("restaurant_id", "")) in selected_restaurant_ids
    ]

    allowed_place_names = _unique_strings(
        [
            *(conditions.get("fixed_order_places") or []),
            *(conditions.get("required_places") or []),
            *(conditions.get("preferred_places") or []),
        ]
    )
    selected_place_names = [
        name
        for name in payload.get("selected_place_names", [])
        if any(
            str(name).casefold() in allowed.casefold()
            or allowed.casefold() in str(name).casefold()
            for allowed in allowed_place_names
        )
        and _document_supports_place(str(name), tourism_documents)
    ]
    if not selected_place_names:
        selected_place_names = [
            name
            for name in allowed_place_names
            if _document_supports_place(name, tourism_documents)
        ]

    places: list[dict[str, Any]] = []
    for restaurant in selected_restaurants:
        if restaurant.get("latitude") is None or restaurant.get("longitude") is None:
            continue
        places.append(
            {
                "restaurant_id": restaurant.get("restaurant_id"),
                "name": restaurant.get("name"),
                "latitude": restaurant.get("latitude"),
                "longitude": restaurant.get("longitude"),
                "category": "restaurant",
            }
        )

    unverified_cost_items = _build_unverified_cost_items(state, payload)
    estimated_total_cost = payload.get("estimated_total_cost")
    total_budget = conditions.get("total_trip_budget")
    budget_exceeded = bool(
        estimated_total_cost is not None
        and total_budget is not None
        and estimated_total_cost > total_budget
    )

    answer = _strip_unsupported_duration_claims(
        str(payload.get("answer") or _grounded_partial_answer(state)),
        state,
    )
    if unverified_cost_items:
        if re.search(r"미확인 비용(?: 항목)?\s*[:：]?\s*없음", answer):
            answer = re.sub(
                r"미확인 비용(?: 항목)?\s*[:：]?\s*없음",
                "미확인 비용 항목: " + ", ".join(unverified_cost_items),
                answer,
            )
        elif "미확인 비용" not in answer:
            answer += "\n\n**미확인 비용 항목**\n" + "\n".join(
                f"- {item}" for item in unverified_cost_items
            )

    warnings_list = _unique_strings(
        [
            *state.get("warnings", []),
            *payload.get("warnings", []),
        ]
    )

    required_supported = all(
        _document_supports_place(place, tourism_documents)
        for place in required_places
    ) if required_places else bool(tourism_documents)
    meal_requests = conditions.get("meal_requests", []) or []
    meals_complete = all(
        state.get("meal_candidates", {}).get(
            str(request.get("request_id") or ""),
            [],
        )
        for request in meal_requests
    ) if meal_requests else True
    degraded_search = any(
        "불안정" in warning or "사용할 수 없어" in warning
        for warning in warnings_list
    )
    search_status = (
        "success"
        if required_supported and meals_complete and not degraded_search
        else "partial"
    )

    result: dict[str, Any] = {
        "answer": answer,
        "itinerary": itinerary,
        "selected_places": [
            {"name": name}
            for name in selected_place_names
        ],
        "selected_restaurants": selected_restaurants,
        "places": places,
        "budget_exceeded": budget_exceeded,
        "budget_calculation_status": (
            "partial"
            if unverified_cost_items
            and any(
                payload.get(key) is not None
                for key in (
                    "estimated_food_cost",
                    "estimated_activity_cost",
                    "estimated_transportation_cost",
                    "estimated_total_cost",
                )
            )
            else "cannot_determine"
            if unverified_cost_items
            else payload.get(
                "budget_calculation_status", "cannot_determine"
            )
        ),
        "budget_estimation_notes": payload.get(
            "budget_estimation_notes", []
        ),
        "unverified_cost_items": unverified_cost_items,
        "schedule_conflicts": payload.get("schedule_conflicts", []),
        "route": [],
        "route_distance_basis": "unknown",
        "warnings": warnings_list,
        "search_status": search_status,
    }

    for key in (
        "estimated_food_cost",
        "estimated_activity_cost",
        "estimated_transportation_cost",
        "estimated_total_cost",
    ):
        if payload.get(key) is not None:
            result[key] = payload[key]

    if total_budget is not None and estimated_total_cost is not None:
        result["remaining_budget"] = total_budget - estimated_total_cost

    return result

def create_planner_graph():
    workflow = StateGraph(PlannerState)

    workflow.add_node("analyze", analyze_query_node)
    workflow.add_node("validate", validate_conditions_node)
    workflow.add_node("retrieve_places", retrieve_places_node)
    workflow.add_node("retrieve_meals", retrieve_meals_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate_plan", generate_plan_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "retrieve_places": "retrieve_places",
            "build_context": "build_context",
        },
    )
    workflow.add_edge("retrieve_places", "retrieve_meals")
    workflow.add_edge("retrieve_meals", "build_context")
    workflow.add_edge("build_context", "generate_plan")
    workflow.add_edge("generate_plan", END)

    return workflow.compile()


graph = create_planner_graph()
graph_app = graph


def run(
    question: str,
    previous_state: PlannerState | None = None,
) -> dict[str, Any]:
    initial_state: PlannerState = {
        "question": question,
        "warnings": [],
        "unsupported_conditions": [],
        "condition_conflicts": [],
        "schedule_conflicts": [],
        "unverified_cost_items": [],
        "meal_failure_reasons": {},
        "meal_budget_evaluations": {},
        "needs_clarification": False,
        "clarification_question": "",
        "search_status": "partial",
        "error": "",
    }

    if previous_state:
        previous_itinerary = previous_state.get(
            "itinerary",
            previous_state.get("previous_itinerary", []),
        )
        initial_state["previous_itinerary"] = previous_itinerary
        initial_state["context"] = json.dumps(
            {
                "__planner_previous_state__": True,
                "previous_question": previous_state.get("question", ""),
                "previous_conditions": previous_state.get("conditions", {}),
                "previous_needs_clarification": previous_state.get(
                    "needs_clarification", False
                ),
                "previous_clarification_question": previous_state.get(
                    "clarification_question", ""
                ),
                "previous_itinerary": previous_itinerary,
            },
            ensure_ascii=False,
            default=str,
        )

    return graph.invoke(initial_state)
