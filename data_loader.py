from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve_project_path(value: str) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환한다."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


DATA_DIR = _resolve_project_path(
    os.getenv("DATA_DIR", "data")
)

RESTAURANTS_FILE = os.getenv(
    "RESTAURANTS_FILE",
    "restaurants_op_merged_20260728.csv",
).strip()

MENUS_FILE = os.getenv(
    "MENUS_FILE",
    "menus_selected_20260727.csv",
).strip()

DATA_ENCODING = os.getenv(
    "DATA_ENCODING",
    "utf-8-sig",
).strip() or "utf-8-sig"


RESTAURANT_REQUIRED_COLUMNS = {
    "RSTR_ID",
    "RSTR_NM",
    "RSTR_RDNMADR",
    "RSTR_LNNO_ADRES",
    "RSTR_LA",
    "RSTR_LO",
    "RSTR_INTRCN_CONT",
    "PRKG_POS_YN",
    "PET_ENTRN_POSBL_YN",
    "FGGG_MENU_OFR_YN",
    "RESTDY_INFO_CN",
    "BSNS_TM_CN",
    "CRCMF_LDMARK_NM",
    "CRCMF_LDMARK_LA",
    "CRCMF_LDMARK_LO",
    "CRCMF_LDMARK_DIST",
    "REPRSNT_MENU_NM",
}

MENU_REQUIRED_COLUMNS = {
    "MENU_ID",
    "MENU_NM",
    "MENU_PRICE",
    "RSTR_ID",
}

RESTAURANT_SEARCH_COLUMNS = (
    "RSTR_NM",
    "RSTR_RDNMADR",
    "RSTR_LNNO_ADRES",
    "RSTR_INTRCN_CONT",
    "CRCMF_LDMARK_NM",
    "REPRSNT_MENU_NM",
)

QUERY_STOPWORDS = {
    "서울",
    "서울에서",
    "식당",
    "음식점",
    "맛집",
    "추천",
    "추천해줘",
    "추천해주세요",
    "알려줘",
    "알려주세요",
    "찾아줘",
    "찾아주세요",
    "근처",
    "주변",
    "가까운",
    "있는",
    "좋은",
    "곳",
    "어디",
    "뭐",
    "먹을",
    "먹기",
    "에서",
    "으로",
    "하고",
    "해줘",
    "해주세요",
}


def _require_file(filename: str) -> Path:
    """설정된 데이터 파일이 실제로 존재하는지 확인한다."""
    if not filename:
        raise RuntimeError("데이터 파일명이 비어 있습니다.")

    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            "데이터 파일을 찾을 수 없습니다: "
            f"{path}\n"
            "DATA_DIR, RESTAURANTS_FILE, MENUS_FILE 설정을 확인해주세요."
        )

    return path


def _detect_separator(path: Path) -> str:
    """CSV 확장자라도 실제 파일의 탭·쉼표 구분자를 헤더로 판별한다."""
    with path.open(
        "r",
        encoding=DATA_ENCODING,
        errors="replace",
    ) as file:
        header = file.readline()

    if "\t" in header:
        return "\t"

    return ","


def _read_table(path: Path) -> pd.DataFrame:
    """현재 CSV/TSV와 이전 XLSX 형식을 모두 읽는다."""
    suffix = path.suffix.casefold()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(
            path,
            sep=_detect_separator(path),
            encoding=DATA_ENCODING,
            low_memory=False,
        )

    raise ValueError(
        f"지원하지 않는 데이터 파일 형식입니다: {path.suffix}"
    )


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    filename: str,
) -> None:
    """Graph와 RAG에서 필요한 열이 빠졌는지 조기에 확인한다."""
    missing = sorted(required_columns - set(dataframe.columns))

    if missing:
        raise ValueError(
            f"{filename}에 필요한 열이 없습니다: {missing}"
        )


def _normalize_id_column(series: pd.Series) -> pd.Series:
    """숫자형으로 읽힌 ID의 .0을 제거하고 문자열 ID로 통일한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def _normalize_input_id(value: str | int | float) -> str:
    """함수 입력으로 전달된 식당·메뉴 ID를 데이터의 문자열 형식에 맞춘다."""
    text = str(value).strip()

    if not text:
        return ""

    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]

    return text


def _safe_value(value: Any) -> Any:
    """pandas 결측값과 numpy 스칼라를 일반 Python 값으로 변환한다."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def _restaurant_to_dict(row: pd.Series) -> dict[str, Any]:
    road_address = _safe_value(row.get("RSTR_RDNMADR"))
    lot_address = _safe_value(row.get("RSTR_LNNO_ADRES"))

    return {
        "restaurant_id": str(row["RSTR_ID"]),
        "name": _safe_value(row.get("RSTR_NM")),
        "address": road_address or lot_address,
        "road_address": road_address,
        "lot_address": lot_address,
        "latitude": _safe_value(row.get("RSTR_LA")),
        "longitude": _safe_value(row.get("RSTR_LO")),
        "introduction": _safe_value(row.get("RSTR_INTRCN_CONT")),
        "parking_available": _safe_value(row.get("PRKG_POS_YN")),
        "pet_allowed": _safe_value(row.get("PET_ENTRN_POSBL_YN")),
        "foreign_menu_available": _safe_value(
            row.get("FGGG_MENU_OFR_YN")
        ),
        "rest_day": _safe_value(row.get("RESTDY_INFO_CN")),
        "business_hours": _safe_value(row.get("BSNS_TM_CN")),
        "landmark": _safe_value(row.get("CRCMF_LDMARK_NM")),
        "landmark_latitude": _safe_value(
            row.get("CRCMF_LDMARK_LA")
        ),
        "landmark_longitude": _safe_value(
            row.get("CRCMF_LDMARK_LO")
        ),
        "landmark_distance": _safe_value(
            row.get("CRCMF_LDMARK_DIST")
        ),
        "representative_menu": _safe_value(
            row.get("REPRSNT_MENU_NM")
        ),
    }


def _menu_to_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "menu_id": str(row["MENU_ID"]),
        "restaurant_id": str(row["RSTR_ID"]),
        "menu_name": _safe_value(row.get("MENU_NM")),
        "menu_price": _safe_value(row.get("MENU_PRICE")),
    }


@lru_cache(maxsize=1)
def load_restaurants() -> pd.DataFrame:
    """병합 완료된 식당 기본·운영 데이터를 한 번만 읽는다."""
    path = _require_file(RESTAURANTS_FILE)
    restaurants = _read_table(path)
    _validate_columns(
        restaurants,
        RESTAURANT_REQUIRED_COLUMNS,
        path.name,
    )

    restaurants = restaurants.copy()
    restaurants["RSTR_ID"] = _normalize_id_column(
        restaurants["RSTR_ID"]
    )
    restaurants = restaurants.dropna(subset=["RSTR_ID"])
    restaurants = restaurants.drop_duplicates(
        subset=["RSTR_ID"],
        keep="first",
    ).reset_index(drop=True)

    numeric_columns = (
        "RSTR_LA",
        "RSTR_LO",
        "CRCMF_LDMARK_LA",
        "CRCMF_LDMARK_LO",
        "CRCMF_LDMARK_DIST",
    )

    for column in numeric_columns:
        restaurants[column] = pd.to_numeric(
            restaurants[column],
            errors="coerce",
        )

    return restaurants


@lru_cache(maxsize=1)
def load_menus() -> pd.DataFrame:
    """메뉴 데이터를 한 번만 읽고 ID와 가격 자료형을 정리한다."""
    path = _require_file(MENUS_FILE)
    menus = _read_table(path)
    _validate_columns(
        menus,
        MENU_REQUIRED_COLUMNS,
        path.name,
    )

    menus = menus.copy()
    menus["RSTR_ID"] = _normalize_id_column(
        menus["RSTR_ID"]
    )
    menus["MENU_ID"] = _normalize_id_column(
        menus["MENU_ID"]
    )
    menus["MENU_PRICE"] = pd.to_numeric(
        menus["MENU_PRICE"],
        errors="coerce",
    )

    menus = menus.dropna(
        subset=["RSTR_ID", "MENU_ID"]
    ).reset_index(drop=True)

    return menus


@lru_cache(maxsize=1)
def _restaurant_search_columns() -> dict[str, pd.Series]:
    """반복 키워드 검색을 위해 식당 문자열 열을 소문자로 캐시한다."""
    restaurants = load_restaurants()
    return {
        column: restaurants[column]
        .fillna("")
        .astype("string")
        .str.casefold()
        for column in RESTAURANT_SEARCH_COLUMNS
    }


@lru_cache(maxsize=1)
def _menu_name_search_series() -> pd.Series:
    """반복 키워드 검색을 위해 메뉴명 문자열을 소문자로 캐시한다."""
    return (
        load_menus()["MENU_NM"]
        .fillna("")
        .astype("string")
        .str.casefold()
    )


def clear_data_cache() -> None:
    """개발 중 파일을 교체한 뒤 모든 데이터 캐시를 초기화한다."""
    _menu_name_search_series.cache_clear()
    _restaurant_search_columns.cache_clear()
    load_restaurants.cache_clear()
    load_menus.cache_clear()


def get_data_info() -> dict[str, Any]:
    """app.py 상태 점검에서 사용할 파일·행 수 정보를 반환한다."""
    restaurants_path = _require_file(RESTAURANTS_FILE)
    menus_path = _require_file(MENUS_FILE)
    restaurants = load_restaurants()
    menus = load_menus()

    restaurant_ids = set(restaurants["RSTR_ID"].dropna().tolist())
    menu_restaurant_ids = set(menus["RSTR_ID"].dropna().tolist())

    return {
        "data_dir": str(DATA_DIR),
        "restaurants_file": str(restaurants_path),
        "menus_file": str(menus_path),
        "restaurant_count": int(len(restaurants)),
        "menu_count": int(len(menus)),
        "menu_restaurant_count": int(menus["RSTR_ID"].nunique()),
        "orphan_menu_restaurant_count": int(
            len(menu_restaurant_ids - restaurant_ids)
        ),
    }


def get_landmark_names(
    keyword: str = "",
    limit: int = 50,
) -> list[str]:
    """데이터에 등록된 랜드마크명을 중복 없이 조회한다."""
    if limit <= 0:
        return []

    landmarks = (
        load_restaurants()["CRCMF_LDMARK_NM"]
        .dropna()
        .astype("string")
        .str.strip()
    )
    landmarks = landmarks[landmarks.ne("")]
    keyword = keyword.strip()

    if keyword:
        landmarks = landmarks[
            landmarks.str.contains(
                keyword,
                case=False,
                na=False,
                regex=False,
            )
        ]

    return (
        landmarks
        .drop_duplicates()
        .head(limit)
        .tolist()
    )


def search_restaurants_by_landmark(
    landmark_name: str,
    limit: int = 20,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """원본 데이터에 등록된 랜드마크명과 거리로 식당을 찾는다."""
    landmark_name = landmark_name.strip()

    if not landmark_name or limit <= 0:
        return []

    restaurants = load_restaurants()
    matched = restaurants[
        restaurants["CRCMF_LDMARK_NM"]
        .fillna("")
        .astype("string")
        .str.contains(
            landmark_name,
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    if max_distance is not None:
        matched = matched[
            matched["CRCMF_LDMARK_DIST"].notna()
            & (
                matched["CRCMF_LDMARK_DIST"]
                <= max_distance
            )
        ]

    matched = matched.sort_values(
        "CRCMF_LDMARK_DIST",
        ascending=True,
        na_position="last",
    ).head(limit)

    return [
        _restaurant_to_dict(row)
        for _, row in matched.iterrows()
    ]


def _tokenize_query(query: str) -> list[str]:
    """자연어 질문에서 데이터 검색에 의미 있는 토큰만 남긴다."""
    tokens = re.findall(
        r"[가-힣A-Za-z0-9]+",
        query,
    )

    result: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        normalized = token.strip().casefold()

        if (
            len(normalized) < 2
            or normalized in QUERY_STOPWORDS
            or normalized in seen
        ):
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def search_restaurants_by_keyword(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    식당명·주소·소개·대표메뉴·전체 메뉴명에서 키워드 후보를 찾는다.

    이 함수는 ChromaDB 검색이 실패하거나 관련 ID가 부족할 때 fallback으로 사용한다.
    """
    query = query.strip()

    if not query or limit <= 0:
        return []

    restaurants = load_restaurants()
    menus = load_menus()
    search_columns = _restaurant_search_columns()
    menu_names = _menu_name_search_series()
    tokens = _tokenize_query(query)

    if not tokens:
        tokens = [query.casefold()]

    restaurant_score = pd.Series(
        0,
        index=restaurants.index,
        dtype="int64",
    )
    column_weights = {
        "RSTR_NM": 8,
        "REPRSNT_MENU_NM": 6,
        "CRCMF_LDMARK_NM": 5,
        "RSTR_RDNMADR": 3,
        "RSTR_LNNO_ADRES": 3,
        "RSTR_INTRCN_CONT": 2,
    }

    for token in tokens:
        for column, weight in column_weights.items():
            mask = search_columns[column].str.contains(
                token,
                na=False,
                regex=False,
            )
            restaurant_score += mask.astype("int64") * weight

    menu_score_by_restaurant: dict[str, int] = {}

    for token in tokens:
        menu_mask = menu_names.str.contains(
            token,
            na=False,
            regex=False,
        )

        if not menu_mask.any():
            continue

        matched_menu_ids = menus.loc[menu_mask, "RSTR_ID"]
        counts = matched_menu_ids.value_counts()

        for restaurant_id, count in counts.items():
            restaurant_id_text = str(restaurant_id)
            menu_score_by_restaurant[restaurant_id_text] = (
                menu_score_by_restaurant.get(restaurant_id_text, 0)
                + min(int(count), 5) * 4
            )

    restaurant_score += (
        restaurants["RSTR_ID"]
        .map(menu_score_by_restaurant)
        .fillna(0)
        .astype("int64")
    )

    matched = restaurants.loc[
        restaurant_score > 0
    ].copy()

    if matched.empty:
        return []

    matched["_match_score"] = restaurant_score.loc[
        matched.index
    ]
    matched = matched.sort_values(
        [
            "_match_score",
            "CRCMF_LDMARK_DIST",
            "RSTR_NM",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        na_position="last",
    ).head(limit)

    results: list[dict[str, Any]] = []

    for index, row in matched.iterrows():
        item = _restaurant_to_dict(row)
        fields = [
            column
            for column in RESTAURANT_SEARCH_COLUMNS
            if any(
                token in str(row.get(column) or "").casefold()
                for token in tokens
            )
        ]

        if str(row["RSTR_ID"]) in menu_score_by_restaurant:
            fields.append("MENU_NM")

        item["keyword_match_score"] = int(row["_match_score"])
        item["keyword_match_fields"] = sorted(set(fields))
        results.append(item)

    return results


def get_restaurants_by_ids(
    restaurant_ids: Iterable[str | int | float],
) -> list[dict[str, Any]]:
    """입력된 순서를 유지하며 식당 ID로 상세정보를 조회한다."""
    normalized_ids = [
        _normalize_input_id(value)
        for value in restaurant_ids
        if value is not None
    ]
    normalized_ids = [value for value in normalized_ids if value]

    if not normalized_ids:
        return []

    restaurants = load_restaurants()
    order = {
        restaurant_id: index
        for index, restaurant_id in enumerate(normalized_ids)
    }
    matched = restaurants[
        restaurants["RSTR_ID"].isin(normalized_ids)
    ].copy()

    if matched.empty:
        return []

    matched["_input_order"] = matched["RSTR_ID"].map(order)
    matched = matched.sort_values("_input_order")

    return [
        _restaurant_to_dict(row)
        for _, row in matched.iterrows()
    ]


def get_menus_by_restaurant_ids(
    restaurant_ids: Iterable[str | int | float],
    min_price: float | None = None,
    max_price: float | None = None,
    limit_per_restaurant: int | None = None,
) -> list[dict[str, Any]]:
    """식당 ID에 연결된 메뉴를 가격순으로 조회한다."""
    normalized_ids = [
        _normalize_input_id(value)
        for value in restaurant_ids
        if value is not None
    ]
    normalized_ids = [value for value in normalized_ids if value]

    if not normalized_ids:
        return []

    menus = load_menus()
    matched = menus[
        menus["RSTR_ID"].isin(normalized_ids)
    ].copy()

    if min_price is not None:
        matched = matched[
            matched["MENU_PRICE"].notna()
            & (matched["MENU_PRICE"] >= min_price)
        ]

    if max_price is not None:
        matched = matched[
            matched["MENU_PRICE"].notna()
            & (matched["MENU_PRICE"] <= max_price)
        ]

    if matched.empty:
        return []

    input_order = {
        restaurant_id: index
        for index, restaurant_id in enumerate(normalized_ids)
    }
    matched["_restaurant_order"] = matched["RSTR_ID"].map(input_order)
    matched = matched.sort_values(
        [
            "_restaurant_order",
            "MENU_PRICE",
            "MENU_NM",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        na_position="last",
    )

    if (
        limit_per_restaurant is not None
        and limit_per_restaurant > 0
    ):
        matched = (
            matched
            .groupby("RSTR_ID", sort=False)
            .head(limit_per_restaurant)
        )

    return [
        _menu_to_dict(row)
        for _, row in matched.iterrows()
    ]
