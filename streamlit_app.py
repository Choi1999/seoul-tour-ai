from __future__ import annotations

import html
import re
from typing import Any, Callable

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="서울 한입여행 AI",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 화면 전체 스타일
st.markdown(
    """
    <style>
    :root {
        --brand: #e85d3f;
        --brand-dark: #bd3f28;
        --brand-soft: #fff2ed;
        --ink: #2f2926;
        --muted: #746a65;
        --line: #eadfd8;
        --card: #fffdfb;
        --success: #147d64;
        --warning: #a65c00;
        --danger: #b42318;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(232, 93, 63, 0.10), transparent 28rem),
            linear-gradient(180deg, #fffaf7 0%, #ffffff 38%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2f2926 0%, #201c1a 100%);
    }

    [data-testid="stSidebar"] * {
        color: #fffaf7;
    }

    [data-testid="stSidebar"] .stButton > button {
        border: 1px solid rgba(255, 255, 255, 0.20);
        background: rgba(255, 255, 255, 0.08);
        color: #fffaf7;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.50);
        background: rgba(255, 255, 255, 0.14);
    }

    .hero {
        padding: 1.35rem 1.45rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: rgba(255, 253, 251, 0.92);
        box-shadow: 0 14px 40px rgba(88, 48, 31, 0.08);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: var(--brand-dark);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(1.75rem, 4vw, 2.65rem);
        font-weight: 900;
        line-height: 1.12;
        margin: 0;
    }

    .hero-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        margin: 0.7rem 0 0;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.3rem 0 0.85rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .status-success {
        color: #0b684f;
        background: #e9f8f3;
        border-color: #bce8d8;
    }

    .status-partial {
        color: #8a4b00;
        background: #fff4df;
        border-color: #f2d6a5;
    }

    .status-no-results,
    .status-failed {
        color: #9a261d;
        background: #fff0ee;
        border-color: #f1c1bc;
    }

    .result-card {
        height: 100%;
        padding: 1rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--card);
        box-shadow: 0 8px 24px rgba(88, 48, 31, 0.06);
        overflow-wrap: anywhere;
    }

    .result-card h4 {
        color: var(--ink);
        font-size: 1.08rem;
        margin: 0 0 0.65rem;
    }

    .result-meta {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0.18rem 0;
    }

    .result-description {
        color: #504844;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.65rem;
        padding-top: 0.65rem;
        border-top: 1px dashed var(--line);
    }

    .mini-badge {
        display: inline-block;
        padding: 0.18rem 0.46rem;
        margin: 0.15rem 0.22rem 0.05rem 0;
        border-radius: 999px;
        background: var(--brand-soft);
        color: var(--brand-dark);
        border: 1px solid #f3cfc3;
        font-size: 0.74rem;
        font-weight: 750;
    }

    .timeline-item {
        position: relative;
        padding: 0.9rem 1rem 0.9rem 1.25rem;
        margin: 0.55rem 0 0.55rem 0.65rem;
        border-left: 3px solid #f0a28f;
        border-radius: 0 16px 16px 0;
        background: #fffdfb;
        box-shadow: 0 6px 18px rgba(88, 48, 31, 0.05);
    }

    .timeline-item::before {
        content: "";
        position: absolute;
        left: -0.48rem;
        top: 1.15rem;
        width: 0.78rem;
        height: 0.78rem;
        border-radius: 50%;
        background: var(--brand);
        border: 3px solid #fffaf7;
    }

    .timeline-title {
        color: var(--ink);
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    .timeline-meta {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .empty-guide {
        padding: 1.25rem;
        border: 1px dashed #dcbcaf;
        border-radius: 18px;
        background: rgba(255, 242, 237, 0.55);
        color: #6d554c;
        line-height: 1.65;
    }

    .footer-note {
        color: #8b7c75;
        font-size: 0.78rem;
        text-align: center;
        margin: 1.7rem 0 0.3rem;
    }

    .stChatMessage {
        border-radius: 18px;
    }

    .stChatInputContainer {
        border-top: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.92);
    }

    div[data-testid="stButton"] > button {
        border-radius: 999px;
        border: 1px solid #e4c7bc;
        color: #6f3a2d;
        background: #fffaf7;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--brand);
        color: var(--brand-dark);
        background: var(--brand-soft);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODE_CONFIG = {
    "restaurant": {
        "label": "맛집 추천",
        "icon": "🍜",
        "title": "서울 맛집을 조건에 맞게 찾아드려요",
        "description": (
            "지역, 메뉴, 가격, 주차, 반려동물 동반 같은 조건을 자연어로 입력하세요. "
            "검색 결과는 현재 프로젝트에 포함된 식당·메뉴 데이터를 기준으로 안내합니다."
        ),
        "placeholder": "예: 종각 근처 3만원 이하 닭한마리 알려줘",
        "examples": [
            "종각 근처 3만원 이하 닭한마리 알려줘",
            "경복궁 주변에서 주차 가능한 한식집 추천해줘",
            "강남역 근처 2만원 이하 피자집 알려줘",
        ],
    },
    "planner": {
        "label": "관광 계획",
        "icon": "🗺️",
        "title": "관광지와 맛집을 묶어 서울 일정을 만들어요",
        "description": (
            "가고 싶은 장소와 식사 위치를 함께 적어 주세요. "
            "실제 경로 API는 사용하지 않으므로 이동 시간은 확정하지 않고 순서 중심으로 안내합니다."
        ),
        "placeholder": "예: 경복궁을 보고 종각에서 점심을 먹는 반나절 일정 짜줘",
        "examples": [
            "경복궁을 보고 종각에서 점심을 먹는 반나절 일정 짜줘",
            "북촌과 인사동을 둘러보고 한식으로 저녁 먹는 일정 짜줘",
            "비 오는 날 실내 중심으로 강남 반나절 일정 추천해줘",
        ],
    },
}

TECHNICAL_WARNING_PATTERNS = (
    "traceback",
    "error executing",
    "internal error",
    "hnsw",
    "segment reader",
    "error finding id",
    "api key",
    "openai",
    "huggingface",
    "hf hub",
    "exception",
)


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "active_mode": "restaurant",
        "restaurant_state": None,
        "planner_state": None,
        "restaurant_messages": [],
        "planner_messages": [],
        "pending_prompt": None,
        "show_developer_details": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def _load_runners() -> dict[str, Callable[..., dict[str, Any]]]:
    """Graph 모듈은 한 번만 불러오고 대화 State는 session_state에서 별도로 관리한다."""
    from planner.planner_graph import run as planner_run
    from restaurant.restaurant_graph import run as restaurant_run

    return {
        "restaurant": restaurant_run,
        "planner": planner_run,
    }


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _unique_text(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)

    return result


def _format_price(value: Any) -> str:
    try:
        return f"{int(float(value)):,}원"
    except (TypeError, ValueError):
        return str(value or "가격 미확인")


def _format_distance(value: Any) -> str:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return ""

    if distance >= 1000:
        return f"직선거리 약 {distance / 1000:.1f}km"
    return f"직선거리 약 {distance:.0f}m"


def _format_yes_no(value: Any) -> str:
    if value is None or value == "":
        return "미확인"

    normalized = str(value).strip().casefold()
    positive = {"y", "yes", "true", "1", "가능", "가능함", "허용"}
    negative = {"n", "no", "false", "0", "불가능", "불가", "미허용"}

    if normalized in positive:
        return "가능"
    if normalized in negative:
        return "불가능"
    return str(value)


def _status_markup(status: str | None) -> str:
    normalized = str(status or "partial").strip().lower()
    labels = {
        "success": ("●", "검색 완료"),
        "partial": ("●", "일부 정보 확인 필요"),
        "no_results": ("●", "검색 결과 없음"),
        "failed": ("●", "검색 처리 실패"),
    }
    icon, label = labels.get(normalized, ("●", "상태 확인 필요"))
    css_status = normalized.replace("_", "-")
    return (
        '<div class="status-row">'
        f'<span class="status-pill status-{css_status}">{icon} {_escape(label)}</span>'
        "</div>"
    )


def _sanitize_warning(warning: str) -> str:
    text = str(warning or "").strip()
    lowered = text.casefold()

    if any(pattern in lowered for pattern in TECHNICAL_WARNING_PATTERNS):
        return "일부 내부 검색 기능이 불안정하여 확인 가능한 데이터로 답변했습니다."

    return text


def _visible_warnings(result: dict[str, Any]) -> list[str]:
    warnings = [
        *result.get("warnings", []),
        *result.get("unsupported_conditions", []),
        *result.get("condition_conflicts", []),
        *result.get("schedule_conflicts", []),
    ]
    return _unique_text([_sanitize_warning(value) for value in warnings])


def _raw_diagnostics(result: dict[str, Any]) -> list[str]:
    values = [
        *result.get("warnings", []),
        result.get("error", ""),
    ]
    return _unique_text(values)


def _menu_map(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    menu_map: dict[str, list[dict[str, Any]]] = {}

    for menu in result.get("filtered_menus", []) or result.get("menus", []):
        restaurant_id = str(
            menu.get("restaurant_id")
            or menu.get("RSTR_ID")
            or ""
        ).strip()
        if restaurant_id:
            menu_map.setdefault(restaurant_id, []).append(menu)

    return menu_map


def _restaurant_card_html(
    restaurant: dict[str, Any],
    menus: list[dict[str, Any]],
) -> str:
    name = _escape(restaurant.get("name") or restaurant.get("RSTR_NM") or "식당명 미확인")
    address = _escape(
        restaurant.get("address")
        or restaurant.get("RSTR_RDNMADR")
        or restaurant.get("RSTR_LNNO_ADRES")
        or "주소 미확인"
    )
    representative_menu = _escape(
        restaurant.get("representative_menu")
        or restaurant.get("REPRSNT_MENU_NM")
        or ""
    )
    business_hours = _escape(
        restaurant.get("business_hours")
        or restaurant.get("BSNS_TM_CN")
        or ""
    )
    rest_day = _escape(
        restaurant.get("rest_day")
        or restaurant.get("RESTDY_INFO_CN")
        or ""
    )
    introduction = _escape(
        restaurant.get("introduction")
        or restaurant.get("RSTR_INTRCN_CONT")
        or ""
    )
    distance = _format_distance(restaurant.get("distance_m"))

    badges = []
    if distance:
        badges.append(distance)
    if restaurant.get("parking_available") is not None:
        badges.append(f"주차 {_format_yes_no(restaurant.get('parking_available'))}")
    if restaurant.get("pet_allowed") is not None:
        badges.append(f"반려동물 {_format_yes_no(restaurant.get('pet_allowed'))}")
    if restaurant.get("foreign_menu_available") is not None:
        badges.append(f"외국어 메뉴 {_format_yes_no(restaurant.get('foreign_menu_available'))}")

    badge_html = "".join(
        f'<span class="mini-badge">{_escape(value)}</span>'
        for value in badges
    )

    menu_lines = []
    for menu in menus[:4]:
        menu_name = _escape(menu.get("menu_name") or menu.get("MENU_NM") or "메뉴")
        menu_price = _format_price(menu.get("menu_price") or menu.get("MENU_PRICE"))
        menu_lines.append(f"<div class='result-meta'>• {menu_name} · {_escape(menu_price)}</div>")

    if not menu_lines and representative_menu:
        menu_lines.append(
            f"<div class='result-meta'><b>대표 메뉴</b> · {representative_menu}</div>"
        )

    business_line = ""
    if business_hours:
        business_line += f"<div class='result-meta'><b>영업시간</b> · {business_hours}</div>"
    if rest_day:
        business_line += f"<div class='result-meta'><b>휴무</b> · {rest_day}</div>"

    intro_html = (
        f'<div class="result-description">{introduction[:260]}</div>'
        if introduction
        else ""
    )

    return f"""
    <div class="result-card">
        <h4>🍽️ {name}</h4>
        <div class="result-meta">📍 {address}</div>
        {badge_html}
        <div style="margin-top:0.55rem;">{''.join(menu_lines)}</div>
        {business_line}
        {intro_html}
    </div>
    """


def _render_restaurant_results(result: dict[str, Any]) -> None:
    restaurants = (
        result.get("filtered_restaurants", [])
        or result.get("selected_restaurants", [])
    )

    if not restaurants:
        return

    st.markdown("#### 추천 식당")
    menu_map = _menu_map(result)

    for index in range(0, min(len(restaurants), 6), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            item_index = index + offset
            if item_index >= min(len(restaurants), 6):
                continue

            restaurant = restaurants[item_index]
            restaurant_id = str(
                restaurant.get("restaurant_id")
                or restaurant.get("RSTR_ID")
                or ""
            )
            with column:
                st.markdown(
                    _restaurant_card_html(
                        restaurant,
                        menu_map.get(restaurant_id, []),
                    ),
                    unsafe_allow_html=True,
                )



def _render_itinerary(result: dict[str, Any]) -> None:
    itinerary = result.get("itinerary", [])
    selected_places = result.get("selected_places", [])
    selected_restaurants = result.get("selected_restaurants", [])

    if itinerary:
        st.markdown("#### 일정 요약")
        for item in itinerary:
            name = _escape(item.get("name") or "일정 항목")
            start_time = str(item.get("start_time") or "").strip()
            end_time = str(item.get("end_time") or "").strip()
            duration = item.get("duration_minutes")
            item_type = str(item.get("item_type") or "place")
            icon = {
                "restaurant": "🍽️",
                "travel": "🚶",
                "break": "☕",
                "place": "📍",
            }.get(item_type, "📍")

            time_parts = []
            if start_time and end_time:
                time_parts.append(f"{start_time}–{end_time}")
            elif start_time:
                time_parts.append(start_time)
            if duration:
                time_parts.append(f"약 {duration}분")

            notes = _unique_text(item.get("notes", []))
            notes_html = " · ".join(_escape(note) for note in notes[:3])
            meta = " · ".join(_escape(value) for value in time_parts)
            if notes_html:
                meta = f"{meta}<br>{notes_html}" if meta else notes_html

            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="timeline-title">{icon} {name}</div>
                    <div class="timeline-meta">{meta or '세부 시간은 확인이 필요합니다.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif selected_places or selected_restaurants:
        st.markdown("#### 추천 동선")
        order = 1
        for place in selected_places:
            name = _escape(place.get("name") or "관광지")
            st.markdown(f"**{order}. 📍 {name}**")
            order += 1
        for restaurant in selected_restaurants:
            name = _escape(restaurant.get("name") or "식당")
            st.markdown(f"**{order}. 🍽️ {name}**")
            order += 1

    if selected_restaurants:
        _render_restaurant_results(result)



def _render_budget(result: dict[str, Any]) -> None:
    budget_keys = [
        ("estimated_food_cost", "예상 식비"),
        ("estimated_activity_cost", "예상 관광비"),
        ("estimated_transportation_cost", "예상 교통비"),
        ("estimated_total_cost", "예상 총비용"),
        ("remaining_budget", "남은 예산"),
    ]
    available = [(key, label) for key, label in budget_keys if result.get(key) is not None]

    if not available:
        return

    st.markdown("#### 예산 요약")
    columns = st.columns(min(len(available), 3))
    for index, (key, label) in enumerate(available):
        columns[index % len(columns)].metric(label, _format_price(result.get(key)))

    if result.get("budget_exceeded"):
        st.warning("현재 계산 가능한 비용 기준으로 총예산을 초과할 수 있습니다.")



def _extract_map_points(result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    candidates = [
        *result.get("places", []),
        *result.get("filtered_restaurants", []),
        *result.get("selected_restaurants", []),
    ]

    for item in candidates:
        latitude = item.get("latitude") or item.get("RSTR_LA")
        longitude = item.get("longitude") or item.get("RSTR_LO")

        try:
            lat = float(latitude)
            lon = float(longitude)
        except (TypeError, ValueError):
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        point = (round(lat, 7), round(lon, 7))
        if point in seen:
            continue
        seen.add(point)
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "name": item.get("name") or "장소",
            }
        )

    return pd.DataFrame(rows)



def _render_map(result: dict[str, Any]) -> None:
    map_data = _extract_map_points(result)
    if map_data.empty:
        return

    st.markdown("#### 위치 지도")
    st.map(map_data[["lat", "lon"]], use_container_width=True)
    st.caption("지도 표시는 데이터에 좌표가 있는 장소만 포함합니다. 이동 경로는 제공하지 않습니다.")



def _render_sources(result: dict[str, Any]) -> None:
    documents = result.get("candidate_places", [])
    if not documents:
        return

    sources = []
    for document in documents:
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        source = metadata.get("source") or document.get("source") if isinstance(document, dict) else None
        if source:
            sources.append(str(source).replace("\\", "/").split("/")[-1])

    sources = _unique_text(sources)
    if not sources:
        return

    with st.expander("관광 정보 출처"):
        for source in sources[:10]:
            st.write(f"- {source}")



def _render_result_details(result: dict[str, Any], mode: str) -> None:
    st.markdown(_status_markup(result.get("search_status")), unsafe_allow_html=True)

    if mode == "restaurant":
        _render_restaurant_results(result)
    else:
        _render_itinerary(result)
        _render_budget(result)
        _render_sources(result)

    _render_map(result)

    warnings = _visible_warnings(result)
    if warnings:
        with st.expander("확인 필요 사항"):
            for warning in warnings:
                st.write(f"- {warning}")

    if st.session_state.show_developer_details:
        diagnostics = _raw_diagnostics(result)
        if diagnostics:
            with st.expander("개발자 진단 정보"):
                for diagnostic in diagnostics:
                    st.code(diagnostic)



def _render_message_history(mode: str) -> None:
    messages = st.session_state[f"{mode}_messages"]

    for message in messages:
        role = message.get("role", "assistant")
        avatar = "🙂" if role == "user" else MODE_CONFIG[mode]["icon"]
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))
            result = message.get("result")
            if role == "assistant" and isinstance(result, dict):
                _render_result_details(result, mode)



def _reset_mode(mode: str) -> None:
    st.session_state[f"{mode}_state"] = None
    st.session_state[f"{mode}_messages"] = []
    st.session_state.pending_prompt = None



def _run_prompt(mode: str, prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    messages_key = f"{mode}_messages"
    state_key = f"{mode}_state"

    st.session_state[messages_key].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=MODE_CONFIG[mode]["icon"]):
        with st.spinner("서울 데이터를 확인하고 있어요..."):
            try:
                runner = _load_runners()[mode]
                result = runner(
                    prompt,
                    previous_state=st.session_state[state_key],
                )
                if not isinstance(result, dict):
                    raise TypeError("Graph 실행 결과가 dict 형식이 아닙니다.")

                st.session_state[state_key] = result
                answer = str(
                    result.get("answer")
                    or result.get("clarification_question")
                    or "답변을 생성하지 못했습니다."
                )

            except Exception as error:
                result = {
                    "answer": "요청을 처리하는 중 문제가 발생했습니다. 설정과 데이터 파일을 확인해 주세요.",
                    "search_status": "failed",
                    "warnings": [],
                    "error": str(error),
                }
                answer = result["answer"]

        st.markdown(answer)
        _render_result_details(result, mode)

    st.session_state[messages_key].append(
        {
            "role": "assistant",
            "content": answer,
            "result": result,
        }
    )



def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🍜 서울 한입여행")
        st.caption("관광지와 식당 데이터를 함께 활용하는 서울 여행 AI")
        st.divider()

        selected_label = st.radio(
            "서비스 선택",
            options=["맛집 추천", "관광 계획"],
            index=0 if st.session_state.active_mode == "restaurant" else 1,
        )
        mode = "restaurant" if selected_label == "맛집 추천" else "planner"
        st.session_state.active_mode = mode

        st.divider()
        st.markdown("##### 현재 제공 범위")
        st.caption(
            "• 서울 식당·메뉴 조건 검색\n\n"
            "• 관광 문서 기반 일정 제안\n\n"
            "• 직선거리 기준 위치 확인\n\n"
            "• 대화 중 이전 결과 참조"
        )

        st.info(
            "실시간 교통·영업 여부, 예약, 리뷰, 로그인 및 일정 저장 기능은 현재 제공하지 않습니다."
        )

        st.session_state.show_developer_details = st.checkbox(
            "개발자 진단 정보 보기",
            value=st.session_state.show_developer_details,
        )

        if st.button("현재 대화 초기화", use_container_width=True):
            _reset_mode(mode)
            st.rerun()

        if st.button("모든 대화 초기화", use_container_width=True):
            _reset_mode("restaurant")
            _reset_mode("planner")
            st.rerun()

        st.divider()
        st.caption("이 서비스의 거리 표시는 실제 도보·차량 경로가 아닌 직선거리일 수 있습니다.")

    return mode



def main() -> None:
    _init_session_state()
    mode = _render_sidebar()
    config = MODE_CONFIG[mode]

    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">SEOUL FOOD & TOUR AI</div>
            <h1 class="hero-title">{config['icon']} {_escape(config['title'])}</h1>
            <p class="hero-description">{_escape(config['description'])}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    messages = st.session_state[f"{mode}_messages"]
    if not messages:
        st.markdown(
            """
            <div class="empty-guide">
                <b>이렇게 질문해 보세요.</b><br>
                위치와 음식 종류, 예산을 한 문장에 함께 적으면 더 정확한 결과를 받을 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        example_columns = st.columns(len(config["examples"]))
        for index, example in enumerate(config["examples"]):
            if example_columns[index].button(
                example,
                key=f"example_{mode}_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = example

    _render_message_history(mode)

    typed_prompt = st.chat_input(config["placeholder"])
    prompt = typed_prompt or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if prompt:
        _run_prompt(mode, prompt)

    st.markdown(
        """
        <div class="footer-note">
            제공되는 정보는 프로젝트 데이터 기준이며, 방문 전 실제 운영 정보 확인이 필요합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
