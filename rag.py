from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
import warnings
from collections import defaultdict
from functools import lru_cache
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from data_loader import (
    get_menus_by_restaurant_ids,
    get_restaurants_by_ids,
    search_restaurants_by_keyword,
    search_restaurants_by_landmark,
)
from embedding import CHROMA_PATH, get_vectorstore


YES_VALUES = {
    "y",
    "yes",
    "true",
    "1",
    "가능",
    "가능함",
    "허용",
}

DEFAULT_LOCATION_RADIUS_M = 1_500.0

FOOD_TERM_ALIASES: dict[str, list[str]] = {
    "치킨": ["치킨", "통닭", "후라이드", "프라이드", "양념치킨", "닭강정"],
    "순대": ["순대", "순댓국", "순대국", "순대국밥"],
    "국밥": ["국밥", "돼지국밥", "소머리국밥", "순대국밥", "콩나물국밥"],
    "중식": ["중식", "중국집", "중국요리", "짜장면", "자장면", "짬뽕", "탕수육", "마라탕"],
    "피자": ["피자", "피자집", "피제리아", "화덕피자"],
    "분식": ["분식", "떡볶이", "김밥", "순대", "튀김"],
    "일식": ["일식", "초밥", "스시", "회", "돈카츠", "돈가스", "우동", "라멘"],
    "한식": ["한식", "백반", "찌개", "전골", "불고기", "비빔밥"],
    "고기": ["고깃집", "고기집", "삼겹살", "돼지고기", "소고기", "갈비", "구이"],
    "카페": ["카페", "커피", "디저트", "브런치", "베이커리"],
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_term(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value)).strip()


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = _normalize_term(value)
        key = text.casefold()

        if not text or key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    return number


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)

    if number is None:
        return None

    return int(number)


def _is_yes(value: Any) -> bool:
    return _clean_text(value).casefold() in YES_VALUES


def _normalized_filename(source: Any) -> str:
    source_text = _clean_text(source).replace("\\", "/")
    return unicodedata.normalize("NFC", Path(source_text).name).casefold()


def _source_kind(metadata: Mapping[str, Any]) -> str:
    source = _clean_text(metadata.get("source")).replace("\\", "/")
    filename = _normalized_filename(source)

    if "menus_selected" in filename:
        return "menu"

    if (
        "restaurants_op_merged" in filename
        or "restaurants_selected" in filename
    ):
        return "restaurant"

    if filename.endswith(".pdf"):
        # 음식점 명단형 PDF는 관광 일정 근거로 사용하지 않는다.
        if "음식점 정보" in filename or "착한가격업소" in filename:
            return "restaurant_reference"
        return "tourism"

    return "unknown"


@lru_cache(maxsize=8)
def _get_chroma_sources(document_kind: str) -> tuple[str, ...]:
    """Chroma SQLite에서 문서 종류별 실제 source 값을 조회한다."""
    sqlite_path = CHROMA_PATH / "chroma.sqlite3"

    if not sqlite_path.is_file():
        return ()

    wanted_kinds = {
        "restaurant_data": {"restaurant", "menu"},
        "restaurant": {"restaurant"},
        "menu": {"menu"},
        "tourism": {"tourism"},
    }.get(document_kind, {document_kind})

    try:
        with sqlite3.connect(sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT string_value
                FROM embedding_metadata
                WHERE key = 'source'
                  AND string_value IS NOT NULL
                """
            ).fetchall()
    except sqlite3.Error:
        return ()

    return tuple(
        source
        for (source,) in rows
        if _source_kind({"source": source}) in wanted_kinds
    )

def _parse_document_ids(
    content: str,
    kind: str,
) -> tuple[str | None, str | None]:
    """Chroma CSVLoader 문서에서 MENU_ID와 RSTR_ID만 안전하게 추출한다."""
    if ":" not in content:
        return None, None

    payload = content.split(":", 1)[1].strip()
    values = payload.split("\t")

    if kind == "restaurant":
        restaurant_id = values[0].strip() if values else ""
        return None, restaurant_id or None

    if kind == "menu":
        menu_id = values[0].strip() if len(values) >= 1 else ""
        restaurant_id = values[3].strip() if len(values) >= 4 else ""
        return menu_id or None, restaurant_id or None

    return None, None


def _document_to_reference(
    document: Any,
    distance: float | None = None,
) -> dict[str, Any]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    content = _clean_text(getattr(document, "page_content", ""))
    kind = _source_kind(metadata)
    menu_id, restaurant_id = _parse_document_ids(content, kind)

    return {
        "kind": kind,
        "source": _clean_text(metadata.get("source")),
        "row": metadata.get("row"),
        "page": metadata.get("page"),
        "content": content,
        "menu_id": menu_id,
        "restaurant_id": restaurant_id,
        "distance": distance,
        "metadata": metadata,
    }


def _deduplicate_documents(
    references: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for reference in references:
        key = (
            reference.get("source"),
            reference.get("row"),
            reference.get("page"),
            reference.get("content"),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(reference)

    return result


def is_vector_db_available() -> bool:
    """기존 ChromaDB의 필수 SQLite 파일 존재 여부만 확인한다."""
    return (
        CHROMA_PATH.is_dir()
        and (CHROMA_PATH / "chroma.sqlite3").is_file()
    )


def search_documents(
    query: str,
    *,
    k: int = 10,
    document_kind: str | None = None,
    fetch_k: int | None = None,
) -> list[dict[str, Any]]:
    """기존 ChromaDB를 조회하고 source 종류를 구분해 반환한다."""
    query = _normalize_term(query)

    if not query or k <= 0:
        return []

    if not is_vector_db_available():
        raise FileNotFoundError(
            f"ChromaDB를 찾을 수 없습니다: {CHROMA_PATH}"
        )

    requested_fetch_k = fetch_k or max(k * 4, 30)
    requested_fetch_k = max(k, min(requested_fetch_k, 200))
    vectorstore = get_vectorstore()

    metadata_filter: dict[str, Any] | None = None
    wanted_kinds: set[str] | None = None

    if document_kind:
        wanted_kinds = {
            "restaurant_data": {"restaurant", "menu"},
        }.get(document_kind, {document_kind})
        sources = _get_chroma_sources(document_kind)
        if sources:
            metadata_filter = {"source": {"$in": list(sources)}}

    search_kwargs: dict[str, Any] = {
        "k": requested_fetch_k,
    }
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    first_error: Exception | None = None

    try:
        pairs = vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )
        references = [
            _document_to_reference(document, _safe_float(distance))
            for document, distance in pairs
        ]
    except Exception as error:
        first_error = error
        try:
            documents = vectorstore.similarity_search(
                query,
                **search_kwargs,
            )
            references = [
                _document_to_reference(document)
                for document in documents
            ]
        except Exception as retry_error:
            raise RuntimeError(
                "ChromaDB 검색을 두 방식으로 시도했지만 실패했습니다. "
                f"첫 오류: {first_error}; 재시도 오류: {retry_error}"
            ) from retry_error

    references = _deduplicate_documents(references)

    if wanted_kinds:
        references = [
            reference
            for reference in references
            if reference.get("kind") in wanted_kinds
        ]

    return references[:k]


def _tourism_query_terms(query: str) -> list[str]:
    stopwords = {
        "서울", "여행", "관광", "일정", "코스", "반나절", "하루",
        "보고", "먹는", "점심", "저녁", "아침", "짜줘", "알려줘",
    }
    terms = re.findall(r"[가-힣A-Za-z0-9]+", _normalize_term(query))
    return _unique_strings(
        term
        for term in terms
        if len(term) >= 2 and term.casefold() not in stopwords
    )


def _search_tourism_documents_lexical(
    query: str,
    *,
    k: int,
) -> list[dict[str, Any]]:
    """벡터 인덱스가 불안정할 때 Chroma SQLite 원문을 직접 검색한다."""
    sqlite_path = CHROMA_PATH / "chroma.sqlite3"
    terms = _tourism_query_terms(query)

    if not sqlite_path.is_file() or not terms:
        return []

    where_parts = ["d.string_value LIKE ?" for _ in terms]
    parameters = [f"%{term}%" for term in terms]
    sql = f"""
        SELECT
            d.id,
            d.string_value AS content,
            s.string_value AS source,
            p.int_value AS page
        FROM embedding_metadata AS d
        JOIN embedding_metadata AS s
          ON s.id = d.id
         AND s.key = 'source'
        LEFT JOIN embedding_metadata AS p
          ON p.id = d.id
         AND p.key = 'page'
        WHERE d.key = 'chroma:document'
          AND lower(s.string_value) LIKE '%.pdf'
          AND ({' OR '.join(where_parts)})
        LIMIT 600
    """

    try:
        with sqlite3.connect(sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, parameters).fetchall()
    except sqlite3.Error:
        return []

    normalized_query = _normalize_term(query).casefold()
    references: list[dict[str, Any]] = []

    for row in rows:
        source = _clean_text(row["source"])
        if _source_kind({"source": source}) != "tourism":
            continue

        content = _clean_text(row["content"])
        content_folded = content.casefold()
        source_name = _normalized_filename(source)
        score = 0

        if normalized_query and normalized_query in content_folded:
            score += 40

        for term in terms:
            term_folded = term.casefold()
            score += min(content_folded.count(term_folded), 8) * 8
            if term_folded in source_name:
                score += 5

        references.append(
            {
                "kind": "tourism",
                "source": source,
                "row": None,
                "page": row["page"],
                "content": content,
                "menu_id": None,
                "restaurant_id": None,
                "distance": None,
                "metadata": {
                    "source": source,
                    "page": row["page"],
                    "retrieval": "sqlite_lexical",
                },
                "lexical_score": score,
            }
        )

    references.sort(
        key=lambda item: (
            -int(item.get("lexical_score") or 0),
            len(_clean_text(item.get("content"))),
        )
    )
    return _deduplicate_documents(references)[:k]

def search_tourism_documents(
    query: str,
    *,
    k: int = 8,
    use_vector: bool = True,
) -> dict[str, Any]:
    """관광 PDF를 검색하고, 벡터 실패 시 실제 저장 원문으로 대체한다."""
    warnings_list: list[str] = []
    vector_documents: list[dict[str, Any]] = []
    vector_failed = False

    if use_vector:
        try:
            vector_documents = search_documents(
                query,
                k=k,
                document_kind="tourism",
                fetch_k=max(k * 3, 24),
            )
        except Exception:
            vector_failed = True
    else:
        vector_failed = True

    lexical_documents: list[dict[str, Any]] = []
    if vector_failed or len(vector_documents) < k:
        lexical_documents = _search_tourism_documents_lexical(
            query,
            k=max(k, 8),
        )

    documents = _deduplicate_documents(
        [*vector_documents, *lexical_documents]
    )[:k]

    if vector_failed and lexical_documents:
        warnings_list.append(
            "관광 문서의 의미 검색이 불안정하여 저장된 PDF 원문 텍스트 검색을 "
            "함께 사용했습니다."
        )
    elif not use_vector and lexical_documents:
        warnings_list.append(
            "관광 문서는 저장된 PDF 원문 텍스트 검색으로 조회했습니다."
        )

    if not documents:
        warnings_list.append(
            "질문과 직접 관련된 관광 문서를 찾지 못했습니다. "
            "관광지 운영시간·요금·소요시간은 확인할 수 없습니다."
        )

    return {
        "documents": documents,
        "warnings": warnings_list,
        "search_mode": (
            "hybrid"
            if vector_documents and lexical_documents
            else "vector"
            if vector_documents
            else "lexical"
            if lexical_documents
            else "none"
        ),
        "vector_failed": vector_failed,
    }

def _merge_restaurants(
    primary: Iterable[dict[str, Any]],
    secondary: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}

    for item in [*primary, *secondary]:
        restaurant_id = _clean_text(item.get("restaurant_id"))

        if not restaurant_id:
            continue

        if restaurant_id not in index_by_id:
            index_by_id[restaurant_id] = len(result)
            result.append(dict(item))
            continue

        existing = result[index_by_id[restaurant_id]]

        for key, value in item.items():
            if key == "rag_evidence":
                evidence = existing.setdefault("rag_evidence", [])
                for entry in value or []:
                    if entry not in evidence:
                        evidence.append(entry)
            elif existing.get(key) in (None, "", []):
                existing[key] = value

    return result


def _verified_landmark_distance(
    restaurant: Mapping[str, Any],
    landmark_name: str,
) -> float | None:
    landmark = _normalize_term(restaurant.get("landmark"))
    if not landmark_name or landmark_name.casefold() not in landmark.casefold():
        return None

    values = (
        _safe_float(restaurant.get("latitude")),
        _safe_float(restaurant.get("longitude")),
        _safe_float(restaurant.get("landmark_latitude")),
        _safe_float(restaurant.get("landmark_longitude")),
    )
    if any(value is None for value in values):
        return None

    restaurant_lat, restaurant_lon, landmark_lat, landmark_lon = values
    return _haversine_distance_m(
        restaurant_lat,
        restaurant_lon,
        landmark_lat,
        landmark_lon,
    )


def search_restaurant_candidates(
    query: str,
    *,
    k: int = 30,
    keyword_query: str | None = None,
    landmark_name: str | None = None,
    use_vector: bool = True,
) -> dict[str, Any]:
    """벡터·랜드마크·키워드 검색을 합쳐 식당 후보를 만든다."""
    warnings_list: list[str] = []
    evidence_by_restaurant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    restaurant_ids: list[str] = []
    vector_failed = False

    if use_vector and query and is_vector_db_available():
        try:
            references = search_documents(
                query,
                k=max(k * 3, 60),
                document_kind="restaurant_data",
                fetch_k=max(k * 4, 80),
            )

            for reference in references:
                restaurant_id = _clean_text(reference.get("restaurant_id"))
                if not restaurant_id:
                    continue
                if restaurant_id not in restaurant_ids:
                    restaurant_ids.append(restaurant_id)
                evidence_by_restaurant[restaurant_id].append(reference)
                if len(restaurant_ids) >= max(k * 2, 80):
                    break
        except Exception:
            vector_failed = True
            warnings_list.append(
                "의미 기반 식당 검색을 사용할 수 없어 원본 데이터의 "
                "랜드마크·키워드 검색을 함께 사용했습니다."
            )

    vector_restaurants = get_restaurants_by_ids(restaurant_ids)
    for restaurant in vector_restaurants:
        restaurant_id = _clean_text(restaurant.get("restaurant_id"))
        restaurant["rag_evidence"] = evidence_by_restaurant.get(
            restaurant_id,
            [],
        )
        restaurant["search_source"] = "vector"

    keyword_text = _normalize_term(keyword_query or query)
    keyword_restaurants: list[dict[str, Any]] = []
    if keyword_text:
        try:
            keyword_restaurants = search_restaurants_by_keyword(
                keyword_text,
                limit=max(k * 3, 120),
            )
            for restaurant in keyword_restaurants:
                restaurant["search_source"] = "keyword"
        except Exception as error:
            warnings_list.append(
                f"키워드 식당 검색에 실패했습니다: {error}"
            )

    landmark_restaurants: list[dict[str, Any]] = []
    landmark_name = _normalize_term(landmark_name)
    if landmark_name:
        try:
            landmark_restaurants = search_restaurants_by_landmark(
                landmark_name,
                limit=max(k * 15, 500),
            )
            for restaurant in landmark_restaurants:
                restaurant["search_source"] = "landmark"
                verified_distance = _verified_landmark_distance(
                    restaurant,
                    landmark_name,
                )
                if verified_distance is not None:
                    restaurant["verified_landmark_distance_m"] = verified_distance

            landmark_restaurants.sort(
                key=lambda item: (
                    item.get("verified_landmark_distance_m") is None,
                    item.get("verified_landmark_distance_m")
                    if item.get("verified_landmark_distance_m") is not None
                    else float("inf"),
                )
            )
        except Exception as error:
            warnings_list.append(
                f"랜드마크 식당 검색에 실패했습니다: {error}"
            )

    # 위치 조건이 있으면 검증 가능한 랜드마크 후보를 먼저 두고,
    # 그다음 의미·키워드 후보를 합친다.
    primary = landmark_restaurants if landmark_name else vector_restaurants
    secondary = (
        [*vector_restaurants, *keyword_restaurants]
        if landmark_name
        else keyword_restaurants
    )
    restaurants = _merge_restaurants(primary, secondary)
    candidate_pool_limit = max(k * 5, 200)
    restaurants = restaurants[:candidate_pool_limit]

    if not restaurants:
        warnings_list.append("검색 가능한 식당 후보를 찾지 못했습니다.")

    return {
        "restaurants": restaurants,
        "restaurant_ids": [
            _clean_text(item.get("restaurant_id"))
            for item in restaurants
            if _clean_text(item.get("restaurant_id"))
        ],
        "warnings": _unique_strings(warnings_list),
        "vector_failed": vector_failed,
    }

def get_restaurant_menus(
    restaurant_ids: Iterable[str | int | float],
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    limit_per_restaurant: int | None = None,
) -> list[dict[str, Any]]:
    """식당 ID에 연결된 실제 메뉴 데이터를 조회한다."""
    return get_menus_by_restaurant_ids(
        restaurant_ids,
        min_price=min_price,
        max_price=max_price,
        limit_per_restaurant=limit_per_restaurant,
    )


def expand_food_terms(
    conditions: Mapping[str, Any],
) -> list[str]:
    """사용자 음식 표현을 검색용 동의어로 확장한다."""
    food = conditions.get("food") or {}
    source_terms = [
        *(food.get("raw_food_terms") or []),
        *(food.get("normalized_categories") or []),
        *(food.get("menu_keywords") or []),
    ]

    expanded: list[str] = []

    for source_term in source_terms:
        term = _normalize_term(source_term)

        if not term:
            continue

        expanded.append(term)
        compact_term = re.sub(r"(집|전문점)$", "", term).strip()

        if compact_term:
            expanded.append(compact_term)

        for category, aliases in FOOD_TERM_ALIASES.items():
            alias_pool = [category, *aliases]

            if any(
                term.casefold() == alias.casefold()
                or term.casefold() in alias.casefold()
                or alias.casefold() in term.casefold()
                for alias in alias_pool
            ):
                expanded.extend(alias_pool)

    return _unique_strings(expanded)


def build_restaurant_query(
    question: str,
    conditions: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """State 조건으로 ChromaDB에 전달할 검색 문장을 만든다."""
    location = conditions.get("location") or {}
    food_terms = expand_food_terms(conditions)
    parts = [
        *(conditions.get("restaurant_names") or []),
        *food_terms,
        *(conditions.get("preference_keywords") or []),
        _clean_text(location.get("text")),
    ]
    parts = _unique_strings(parts)

    return (" ".join(parts) or _normalize_term(question), food_terms)


def _haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(value), math.sqrt(1 - value))



def _location_terms(value: Any) -> list[str]:
    text = _normalize_term(value)
    if not text:
        return []

    terms = [text]
    compact = re.sub(r"\s*(?:근처|주변|인근)$", "", text).strip()
    if compact:
        terms.append(compact)
    if compact.endswith("역") and len(compact) > 1:
        terms.append(compact[:-1])
    elif compact and not compact.endswith("역"):
        terms.append(f"{compact}역")
    return _unique_strings(terms)


def _match_restaurant_location(
    restaurant: Mapping[str, Any],
    *,
    location_text: str,
    location_latitude: float | None,
    location_longitude: float | None,
) -> tuple[bool, float | None, str]:
    """위치 조건을 원본 주소·랜드마크·좌표로 검증한다."""
    restaurant_latitude = _safe_float(restaurant.get("latitude"))
    restaurant_longitude = _safe_float(restaurant.get("longitude"))

    if location_latitude is not None and location_longitude is not None:
        if restaurant_latitude is None or restaurant_longitude is None:
            return False, None, "unverified"
        return (
            True,
            _haversine_distance_m(
                location_latitude,
                location_longitude,
                restaurant_latitude,
                restaurant_longitude,
            ),
            "straight_line",
        )

    terms = _location_terms(location_text)
    if not terms:
        return True, None, "none"

    address_text = " ".join(
        _clean_text(restaurant.get(key))
        for key in ("address", "road_address", "lot_address")
    )
    landmark_text = _normalize_term(restaurant.get("landmark"))

    if any(_contains_term(address_text, term) for term in terms):
        return True, None, "address_text"

    if any(_contains_term(landmark_text, term) for term in terms):
        landmark_latitude = _safe_float(restaurant.get("landmark_latitude"))
        landmark_longitude = _safe_float(restaurant.get("landmark_longitude"))
        if (
            restaurant_latitude is not None
            and restaurant_longitude is not None
            and landmark_latitude is not None
            and landmark_longitude is not None
        ):
            return (
                True,
                _haversine_distance_m(
                    landmark_latitude,
                    landmark_longitude,
                    restaurant_latitude,
                    restaurant_longitude,
                ),
                "straight_line",
            )
        return True, None, "landmark_text"

    return False, None, "unverified"

def _menus_by_restaurant(
    menus: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for menu in menus:
        restaurant_id = _clean_text(menu.get("restaurant_id"))

        if restaurant_id:
            grouped[restaurant_id].append(dict(menu))

    return grouped


def _contains_term(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def _restaurant_search_text(
    restaurant: Mapping[str, Any],
    menus: Iterable[Mapping[str, Any]],
) -> str:
    values = [
        restaurant.get("name"),
        restaurant.get("address"),
        restaurant.get("road_address"),
        restaurant.get("lot_address"),
        restaurant.get("introduction"),
        restaurant.get("representative_menu"),
        restaurant.get("landmark"),
        *[menu.get("menu_name") for menu in menus],
        *[
            evidence.get("content")
            for evidence in restaurant.get("rag_evidence", []) or []
        ],
    ]
    return " ".join(_clean_text(value) for value in values if _clean_text(value))


def _minimum_menu_price(menus: Iterable[Mapping[str, Any]]) -> float | None:
    prices = [
        price
        for price in (_safe_float(menu.get("menu_price")) for menu in menus)
        if price is not None
    ]
    return min(prices) if prices else None


def filter_restaurant_candidates(
    restaurants: Iterable[dict[str, Any]],
    menus: Iterable[dict[str, Any]],
    conditions: Mapping[str, Any],
    *,
    expanded_search_terms: Iterable[str] = (),
) -> dict[str, Any]:
    """가격·편의·메뉴·위치처럼 정확히 판정 가능한 조건을 적용한다."""
    restaurant_list = [dict(item) for item in restaurants]
    all_menus = [dict(item) for item in menus]
    menu_map = _menus_by_restaurant(all_menus)

    failure_counts: dict[str, int] = defaultdict(int)
    warnings_list: list[str] = []
    category_matches: list[dict[str, Any]] = []
    restaurant_distances: dict[str, float] = {}
    default_radius_used = False

    restaurant_names = _unique_strings(conditions.get("restaurant_names") or [])
    excluded_food_terms = _unique_strings(conditions.get("excluded_food_terms") or [])
    excluded_restaurant_terms = _unique_strings(
        conditions.get("excluded_restaurant_keywords") or []
    )
    excluded_areas = _unique_strings(conditions.get("excluded_areas") or [])

    food = conditions.get("food") or {}
    menu_keywords = _unique_strings(food.get("menu_keywords") or [])
    broad_food_terms = _unique_strings(expanded_search_terms)

    min_price = _safe_float(conditions.get("min_menu_price"))
    max_price = _safe_float(conditions.get("max_menu_price"))

    location = conditions.get("location") or {}
    location_text = _normalize_term(location.get("text"))
    location_latitude = _safe_float(location.get("latitude"))
    location_longitude = _safe_float(location.get("longitude"))
    max_distance_m = _safe_float(conditions.get("max_distance_m"))

    filtered_restaurants: list[dict[str, Any]] = []
    filtered_menus: list[dict[str, Any]] = []

    for restaurant in restaurant_list:
        restaurant_id = _clean_text(restaurant.get("restaurant_id"))
        restaurant_menus = menu_map.get(restaurant_id, [])
        search_text = _restaurant_search_text(restaurant, restaurant_menus)
        name = _clean_text(restaurant.get("name"))
        address_text = " ".join(
            _clean_text(restaurant.get(key))
            for key in ("address", "road_address", "lot_address", "landmark")
        )

        if restaurant_names and not any(
            _contains_term(name, restaurant_name)
            for restaurant_name in restaurant_names
        ):
            failure_counts["restaurant_name"] += 1
            continue

        if excluded_food_terms and any(
            _contains_term(search_text, term)
            for term in excluded_food_terms
        ):
            failure_counts["excluded_food"] += 1
            continue

        if excluded_restaurant_terms and any(
            _contains_term(search_text, term)
            for term in excluded_restaurant_terms
        ):
            failure_counts["excluded_restaurant"] += 1
            continue

        if excluded_areas and any(
            _contains_term(address_text, area)
            for area in excluded_areas
        ):
            failure_counts["excluded_area"] += 1
            continue

        matched_fields: list[str] = []
        matched_values: list[str] = []

        if broad_food_terms:
            for term in broad_food_terms:
                if _contains_term(search_text, term):
                    matched_fields.append("restaurant_or_menu_text")
                    matched_values.append(term)

            if not matched_values:
                failure_counts["food"] += 1
                continue

            category_matches.append(
                {
                    "restaurant_id": restaurant_id,
                    "requested_term": ", ".join(
                        food.get("raw_food_terms") or broad_food_terms
                    ),
                    "matched_category": ", ".join(
                        food.get("normalized_categories") or broad_food_terms
                    ),
                    "matched_fields": _unique_strings(matched_fields),
                    "matched_values": _unique_strings(matched_values),
                    "confidence": "high" if menu_keywords else "medium",
                }
            )

        relevant_menu_rows = restaurant_menus

        if menu_keywords:
            relevant_menu_rows = [
                menu
                for menu in restaurant_menus
                if any(
                    _contains_term(
                        _clean_text(menu.get("menu_name")),
                        keyword,
                    )
                    for keyword in menu_keywords
                )
            ]

            if not relevant_menu_rows:
                failure_counts["menu_keyword"] += 1
                continue

        elif broad_food_terms:
            food_menu_rows = [
                menu
                for menu in restaurant_menus
                if any(
                    _contains_term(
                        _clean_text(menu.get("menu_name")),
                        term,
                    )
                    for term in broad_food_terms
                )
            ]

            # 식당명·대표메뉴·소개에서만 음식 종류가 확인된 경우에는
            # 관련 없는 저가 메뉴를 예산 근거로 사용하지 않도록 메뉴 목록을 비운다.
            relevant_menu_rows = food_menu_rows

        matched_menu_rows = relevant_menu_rows

        if min_price is not None or max_price is not None:
            priced_menu_rows: list[dict[str, Any]] = []

            for menu in matched_menu_rows:
                price = _safe_float(menu.get("menu_price"))

                if price is None:
                    continue

                if min_price is not None and price < min_price:
                    continue

                if max_price is not None and price > max_price:
                    continue

                priced_menu_rows.append(menu)

            matched_menu_rows = priced_menu_rows

            if not matched_menu_rows:
                failure_counts["menu_price"] += 1
                continue

        if conditions.get("parking_required") is True and not _is_yes(
            restaurant.get("parking_available")
        ):
            failure_counts["parking"] += 1
            continue

        if conditions.get("pet_allowed_required") is True and not _is_yes(
            restaurant.get("pet_allowed")
        ):
            failure_counts["pet"] += 1
            continue

        if conditions.get("foreign_menu_required") is True and not _is_yes(
            restaurant.get("foreign_menu_available")
        ):
            failure_counts["foreign_menu"] += 1
            continue

        distance: float | None = None
        distance_basis = "none"

        if location_text or (
            location_latitude is not None and location_longitude is not None
        ):
            location_verified, distance, distance_basis = _match_restaurant_location(
                restaurant,
                location_text=location_text,
                location_latitude=location_latitude,
                location_longitude=location_longitude,
            )

            if not location_verified:
                failure_counts["location"] += 1
                continue

            if distance is not None:
                restaurant_distances[restaurant_id] = distance
                effective_max_distance = max_distance_m
                if effective_max_distance is None:
                    effective_max_distance = DEFAULT_LOCATION_RADIUS_M
                    default_radius_used = True

                if distance > effective_max_distance:
                    failure_counts["distance"] += 1
                    continue

        if conditions.get("visit_day") or conditions.get("visit_time"):
            warnings_list.append(
                "영업시간과 휴무일은 자유 형식 데이터이므로 현재 영업 여부를 "
                "자동으로 단정하지 않고 원문을 제공합니다."
            )

        restaurant["distance_m"] = distance
        if distance is not None:
            restaurant["distance_basis"] = "straight_line"
        elif distance_basis in {"address_text", "landmark_text"}:
            restaurant["distance_basis"] = distance_basis
        restaurant.pop("landmark_distance", None)
        restaurant["menus"] = matched_menu_rows
        filtered_restaurants.append(restaurant)
        filtered_menus.extend(matched_menu_rows)

    if default_radius_used:
        warnings_list.append(
            "수치 거리 없이 위치만 지정되어 서비스 기본 기준인 직선거리 "
            "1.5km 이내로 위치를 검증했습니다."
        )

    sort_by = _clean_text(conditions.get("sort_by")) or "relevance"

    if sort_by == "distance":
        filtered_restaurants.sort(
            key=lambda item: (
                item.get("distance_m") is None,
                item.get("distance_m")
                if item.get("distance_m") is not None
                else float("inf"),
            )
        )
    elif sort_by == "price_low":
        filtered_restaurants.sort(
            key=lambda item: (
                _minimum_menu_price(item.get("menus", [])) is None,
                _minimum_menu_price(item.get("menus", []))
                if _minimum_menu_price(item.get("menus", [])) is not None
                else float("inf"),
            )
        )
    elif sort_by == "price_high":
        filtered_restaurants.sort(
            key=lambda item: (
                _minimum_menu_price(item.get("menus", [])) is not None,
                _minimum_menu_price(item.get("menus", []))
                if _minimum_menu_price(item.get("menus", [])) is not None
                else float("-inf"),
            ),
            reverse=True,
        )
    elif sort_by == "name":
        filtered_restaurants.sort(
            key=lambda item: _clean_text(item.get("name")).casefold()
        )

    limit = _safe_int(conditions.get("limit")) or 5
    limit = max(1, min(limit, 20))
    filtered_restaurants = filtered_restaurants[:limit]
    filtered_ids = {
        _clean_text(item.get("restaurant_id"))
        for item in filtered_restaurants
    }
    filtered_menus = [
        menu
        for menu in filtered_menus
        if _clean_text(menu.get("restaurant_id")) in filtered_ids
    ]

    failure_labels = {
        "restaurant_name": "요청한 식당명",
        "food": "음식 종류",
        "menu_keyword": "메뉴명",
        "menu_price": "메뉴 가격",
        "parking": "주차 가능",
        "pet": "반려동물 출입 가능",
        "foreign_menu": "외국어 메뉴 제공",
        "distance": "최대 거리",
        "location": "위치",
        "excluded_food": "제외 음식",
        "excluded_restaurant": "제외 식당 조건",
        "excluded_area": "제외 지역",
    }
    failure_reasons = [
        f"{failure_labels[key]} 조건에서 {count}개 후보가 제외되었습니다."
        for key, count in failure_counts.items()
        if count > 0
    ]

    return {
        "filtered_restaurants": filtered_restaurants,
        "filtered_menus": filtered_menus,
        "restaurant_distances": restaurant_distances,
        "category_matches": category_matches,
        "filter_failure_reasons": failure_reasons,
        "warnings": _unique_strings(warnings_list),
    }


def search_restaurants_by_conditions(
    question: str,
    conditions: Mapping[str, Any],
    *,
    candidate_limit: int = 50,
    use_vector: bool = True,
) -> dict[str, Any]:
    """Restaurant와 Planner가 함께 사용하는 식당 검색 전체 파이프라인."""
    semantic_query, expanded_terms = build_restaurant_query(question, conditions)
    location = conditions.get("location") or {}
    landmark_name = _clean_text(location.get("text"))

    candidate_bundle = search_restaurant_candidates(
        semantic_query,
        k=max(candidate_limit, 1),
        keyword_query=question,
        landmark_name=landmark_name,
        use_vector=use_vector,
    )
    candidate_restaurants = candidate_bundle["restaurants"]
    restaurant_ids = candidate_bundle["restaurant_ids"]

    try:
        candidate_menus = get_restaurant_menus(restaurant_ids)
    except Exception as error:
        candidate_menus = []
        candidate_bundle["warnings"].append(
            f"후보 식당의 메뉴 데이터를 불러오지 못했습니다: {error}"
        )

    filtered_bundle = filter_restaurant_candidates(
        candidate_restaurants,
        candidate_menus,
        conditions,
        expanded_search_terms=expanded_terms,
    )

    filtered_restaurants = filtered_bundle.get("filtered_restaurants", [])
    search_status = "no_results"
    if filtered_restaurants:
        search_status = (
            "partial" if candidate_bundle.get("vector_failed") else "success"
        )

    return {
        "semantic_query": semantic_query,
        "expanded_search_terms": expanded_terms,
        "candidate_restaurants": candidate_restaurants,
        "candidate_restaurant_ids": restaurant_ids,
        "candidate_menus": candidate_menus,
        **filtered_bundle,
        "warnings": _unique_strings(
            [
                *candidate_bundle.get("warnings", []),
                *filtered_bundle.get("warnings", []),
            ]
        ),
        "search_status": search_status,
    }


def _extract_servings(menu_name: str) -> int | None:
    match = re.search(r"(\d+)\s*인분", menu_name)

    if not match:
        return None

    return int(match.group(1))


def evaluate_restaurant_budgets(
    restaurants: Iterable[Mapping[str, Any]],
    menus: Iterable[Mapping[str, Any]],
    *,
    budget_amount: int,
    people: int = 1,
    budget_scope: str = "group_total",
) -> list[dict[str, Any]]:
    """메뉴 가격과 메뉴명에 적힌 인분 정보만으로 보수적으로 예산을 평가한다."""
    if budget_amount <= 0:
        return []

    people = max(people, 1)
    effective_budget = budget_amount

    if budget_scope == "per_person":
        effective_budget = budget_amount * people

    menu_map = _menus_by_restaurant(menus)
    evaluations: list[dict[str, Any]] = []

    for restaurant in restaurants:
        restaurant_id = _clean_text(restaurant.get("restaurant_id"))
        restaurant_menus = menu_map.get(restaurant_id, [])
        priced_menus = []

        for menu in restaurant_menus:
            price = _safe_int(menu.get("menu_price"))

            if price is None:
                continue

            item = dict(menu)
            item["menu_price"] = price
            item["servings"] = _extract_servings(
                _clean_text(menu.get("menu_name"))
            )
            priced_menus.append(item)

        if not priced_menus:
            evaluations.append(
                {
                    "restaurant_id": restaurant_id,
                    "budget_amount": effective_budget,
                    "people": people,
                    "selected_menu_ids": [],
                    "selected_menus": [],
                    "estimated_total_price": 0,
                    "budget_status": "cannot_determine",
                    "serving_basis": "unknown",
                    "confidence": "low",
                    "notes": ["확인 가능한 숫자형 메뉴 가격이 없습니다."],
                }
            )
            continue

        explicit_options = [
            menu
            for menu in priced_menus
            if menu.get("servings") is not None
            and menu["servings"] >= people
        ]
        explicit_options.sort(key=lambda item: item["menu_price"])

        if explicit_options:
            selected = explicit_options[0]
            within_budget = selected["menu_price"] <= effective_budget
            evaluations.append(
                {
                    "restaurant_id": restaurant_id,
                    "budget_amount": effective_budget,
                    "people": people,
                    "selected_menu_ids": [_clean_text(selected.get("menu_id"))],
                    "selected_menus": [selected],
                    "estimated_total_price": selected["menu_price"],
                    "budget_status": (
                        "within_budget" if within_budget else "over_budget"
                    ),
                    "serving_basis": "explicit",
                    "confidence": "high",
                    "notes": [
                        "메뉴명에 표시된 인분 수를 기준으로 판단했습니다."
                    ],
                }
            )
            continue

        inferred_options = [
            menu
            for menu in priced_menus
            if re.search(
                r"(?:대|大|라지|large|세트|set)",
                _clean_text(menu.get("menu_name")),
                flags=re.IGNORECASE,
            )
        ]
        inferred_options.sort(key=lambda item: item["menu_price"])

        if inferred_options:
            selected = inferred_options[0]
            evaluations.append(
                {
                    "restaurant_id": restaurant_id,
                    "budget_amount": effective_budget,
                    "people": people,
                    "selected_menu_ids": [_clean_text(selected.get("menu_id"))],
                    "selected_menus": [selected],
                    "estimated_total_price": selected["menu_price"],
                    "budget_status": (
                        "possibly_within_budget"
                        if selected["menu_price"] <= effective_budget
                        else "over_budget"
                    ),
                    "serving_basis": "inferred_from_menu_name",
                    "confidence": "medium",
                    "notes": [
                        "메뉴명의 크기 또는 세트 표현으로 여러 명이 먹을 가능성만 "
                        "추정했으며 실제 제공량은 확인되지 않았습니다."
                    ],
                }
            )
            continue

        affordable = sorted(
            [
                menu
                for menu in priced_menus
                if menu["menu_price"] <= effective_budget
            ],
            key=lambda item: item["menu_price"],
        )[:5]

        evaluations.append(
            {
                "restaurant_id": restaurant_id,
                "budget_amount": effective_budget,
                "people": people,
                "selected_menu_ids": [
                    _clean_text(menu.get("menu_id"))
                    for menu in affordable
                ],
                "selected_menus": affordable,
                "estimated_total_price": (
                    affordable[0]["menu_price"] if affordable else 0
                ),
                "budget_status": (
                    "possibly_within_budget" if affordable else "over_budget"
                ),
                "serving_basis": "unknown",
                "confidence": "low",
                "notes": [
                    "예산 안의 메뉴를 제시하지만 인분 정보가 없어 "
                    f"{people}명이 충분히 먹을 수 있다고 단정할 수 없습니다."
                ],
            }
        )

    return evaluations
