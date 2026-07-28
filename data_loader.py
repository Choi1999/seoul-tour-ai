from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")

RESTAURANTS_FILE = os.getenv(
    "RESTAURANTS_FILE",
    "restaurants_selected_20260727.xlsx",
)

RESTAURANTS_OP_FILE = os.getenv(
    "RESTAURANTS_OP_FILE",
    "restaurants_op_selected_20260727.xlsx",
)

MENUS_FILE = os.getenv(
    "MENUS_FILE",
    "menus_selected_20260727.xlsx",
)


def _require_file(filename: str) -> Path:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "data/README.md를 확인하고 엑셀 파일을 data 폴더에 넣어주세요."
        )

    return path


def _normalize_id_column(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="raise")
        .astype("Int64")
        .astype("string")
    )


def _normalize_input_id(value: str | int | float) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def _safe_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _restaurant_to_dict(row: pd.Series) -> dict:
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


def _menu_to_dict(row: pd.Series) -> dict:
    return {
        "menu_id": str(row["MENU_ID"]),
        "restaurant_id": str(row["RSTR_ID"]),
        "menu_name": _safe_value(row.get("MENU_NM")),
        "menu_price": _safe_value(row.get("MENU_PRICE")),
    }


@lru_cache(maxsize=1)
def load_restaurants() -> pd.DataFrame:
    basic = pd.read_excel(
        _require_file(RESTAURANTS_FILE)
    )

    operation = pd.read_excel(
        _require_file(RESTAURANTS_OP_FILE)
    )

    basic["RSTR_ID"] = _normalize_id_column(
        basic["RSTR_ID"]
    )

    operation["RSTR_ID"] = _normalize_id_column(
        operation["RSTR_ID"]
    )

    restaurants = basic.merge(
        operation,
        on="RSTR_ID",
        how="left",
        validate="one_to_one",
    )

    restaurants["CRCMF_LDMARK_DIST"] = pd.to_numeric(
        restaurants["CRCMF_LDMARK_DIST"],
        errors="coerce",
    )

    return restaurants


@lru_cache(maxsize=1)
def load_menus() -> pd.DataFrame:
    menus = pd.read_excel(
        _require_file(MENUS_FILE)
    )

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

    return menus


def clear_data_cache() -> None:
    load_restaurants.cache_clear()
    load_menus.cache_clear()


def get_landmark_names(
    keyword: str = "",
    limit: int = 50,
) -> list[str]:
    restaurants = load_restaurants()

    landmarks = (
        restaurants["CRCMF_LDMARK_NM"]
        .dropna()
        .astype("string")
    )

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
) -> list[dict]:
    landmark_name = landmark_name.strip()

    if not landmark_name or limit <= 0:
        return []

    restaurants = load_restaurants()

    matched = restaurants[
        restaurants["CRCMF_LDMARK_NM"]
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


def search_restaurants_by_keyword(
    query: str,
    limit: int = 20,
) -> list[dict]:
    query = query.strip()

    if not query or limit <= 0:
        return []

    restaurants = load_restaurants()

    searchable_columns = [
        "RSTR_NM",
        "RSTR_RDNMADR",
        "RSTR_LNNO_ADRES",
        "RSTR_INTRCN_CONT",
        "CRCMF_LDMARK_NM",
        "REPRSNT_MENU_NM",
    ]

    search_text = (
        restaurants[searchable_columns]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
    )

    keywords = [
        word
        for word in query.split()
        if word
    ]

    score = pd.Series(
        0,
        index=restaurants.index,
        dtype="int64",
    )

    for keyword in keywords:
        score += search_text.str.contains(
            keyword,
            case=False,
            na=False,
            regex=False,
        ).astype("int64")

    matched = restaurants.loc[
        score > 0
    ].copy()

    matched["_match_score"] = score.loc[
        score > 0
    ]

    matched = matched.sort_values(
        [
            "_match_score",
            "CRCMF_LDMARK_DIST",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    ).head(limit)

    return [
        _restaurant_to_dict(row)
        for _, row in matched.iterrows()
    ]


def get_restaurants_by_ids(
    restaurant_ids: Iterable[str | int | float],
) -> list[dict]:
    normalized_ids = [
        _normalize_input_id(value)
        for value in restaurant_ids
        if value is not None
    ]

    if not normalized_ids:
        return []

    restaurants = load_restaurants()

    order = {
        restaurant_id: index
        for index, restaurant_id
        in enumerate(normalized_ids)
    }

    matched = restaurants[
        restaurants["RSTR_ID"].isin(
            normalized_ids
        )
    ].copy()

    matched["_input_order"] = (
        matched["RSTR_ID"].map(order)
    )

    matched = matched.sort_values(
        "_input_order"
    )

    return [
        _restaurant_to_dict(row)
        for _, row in matched.iterrows()
    ]


def get_menus_by_restaurant_ids(
    restaurant_ids: Iterable[str | int | float],
    min_price: float | None = None,
    max_price: float | None = None,
    limit_per_restaurant: int | None = None,
) -> list[dict]:
    normalized_ids = [
        _normalize_input_id(value)
        for value in restaurant_ids
        if value is not None
    ]

    if not normalized_ids:
        return []

    menus = load_menus()

    matched = menus[
        menus["RSTR_ID"].isin(
            normalized_ids
        )
    ].copy()

    if min_price is not None:
        matched = matched[
            matched["MENU_PRICE"].notna()
            & (
                matched["MENU_PRICE"]
                >= min_price
            )
        ]

    if max_price is not None:
        matched = matched[
            matched["MENU_PRICE"].notna()
            & (
                matched["MENU_PRICE"]
                <= max_price
            )
        ]

    matched = matched.sort_values(
        [
            "RSTR_ID",
            "MENU_PRICE",
        ],
        ascending=[
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
            .groupby(
                "RSTR_ID",
                sort=False,
            )
            .head(limit_per_restaurant)
        )

    return [
        _menu_to_dict(row)
        for _, row in matched.iterrows()
    ]