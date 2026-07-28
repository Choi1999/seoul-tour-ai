from __future__ import annotations

import argparse
import os
import shutil
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from data_loader import load_restaurants

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PERSIST_DIR = (
    BASE_DIR
    / os.getenv(
        "CHROMA_PERSIST_DIR",
        "chroma_db",
    )
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "restaurants",
)


def _text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def _metadata_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def _row_to_document(
    row: pd.Series,
) -> Document:
    fields = [
        ("식당명", row.get("RSTR_NM")),
        (
            "도로명주소",
            row.get("RSTR_RDNMADR"),
        ),
        (
            "지번주소",
            row.get("RSTR_LNNO_ADRES"),
        ),
        (
            "소개",
            row.get("RSTR_INTRCN_CONT"),
        ),
        (
            "인근 랜드마크",
            row.get("CRCMF_LDMARK_NM"),
        ),
        (
            "랜드마크 거리",
            row.get("CRCMF_LDMARK_DIST"),
        ),
        (
            "영업시간",
            row.get("BSNS_TM_CN"),
        ),
        (
            "휴무일",
            row.get("RESTDY_INFO_CN"),
        ),
        (
            "대표 메뉴",
            row.get("REPRSNT_MENU_NM"),
        ),
        (
            "주차 가능 여부",
            row.get("PRKG_POS_YN"),
        ),
        (
            "반려동물 동반 가능 여부",
            row.get("PET_ENTRN_POSBL_YN"),
        ),
        (
            "외국어 메뉴 제공 여부",
            row.get("FGGG_MENU_OFR_YN"),
        ),
    ]

    page_content = "\n".join(
        f"{label}: {_text(value)}"
        for label, value in fields
        if _text(value)
    )

    road_address = _metadata_value(
        row.get("RSTR_RDNMADR")
    )

    lot_address = _metadata_value(
        row.get("RSTR_LNNO_ADRES")
    )

    raw_metadata = {
        "restaurant_id": str(
            row["RSTR_ID"]
        ),
        "name": _metadata_value(
            row.get("RSTR_NM")
        ),
        "address": (
            road_address
            or lot_address
        ),
        "latitude": _metadata_value(
            row.get("RSTR_LA")
        ),
        "longitude": _metadata_value(
            row.get("RSTR_LO")
        ),
        "landmark": _metadata_value(
            row.get("CRCMF_LDMARK_NM")
        ),
        "landmark_distance": (
            _metadata_value(
                row.get(
                    "CRCMF_LDMARK_DIST"
                )
            )
        ),
        "business_hours": (
            _metadata_value(
                row.get("BSNS_TM_CN")
            )
        ),
        "parking_available": (
            _metadata_value(
                row.get("PRKG_POS_YN")
            )
        ),
        "pet_allowed": (
            _metadata_value(
                row.get(
                    "PET_ENTRN_POSBL_YN"
                )
            )
        ),
        "foreign_menu_available": (
            _metadata_value(
                row.get(
                    "FGGG_MENU_OFR_YN"
                )
            )
        ),
    }

    metadata = {
        key: value
        for key, value
        in raw_metadata.items()
        if value is not None
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-m3",
        ),
        model_kwargs={
            "device": os.getenv(
                "EMBEDDING_DEVICE",
                "cpu",
            ),
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=(
            CHROMA_COLLECTION_NAME
        ),
        persist_directory=str(
            CHROMA_PERSIST_DIR
        ),
        embedding_function=(
            get_embeddings()
        ),
    )


def build_vectorstore(
    rebuild: bool = False,
    batch_size: int = 500,
    limit: int | None = None,
) -> Chroma:
    if (
        rebuild
        and CHROMA_PERSIST_DIR.exists()
    ):
        shutil.rmtree(
            CHROMA_PERSIST_DIR
        )

    vectorstore = get_vectorstore()

    existing_count = (
        vectorstore
        ._collection
        .count()
    )

    if existing_count > 0 and not rebuild:
        print(
            f"기존 ChromaDB에 "
            f"{existing_count:,}개 문서가 있습니다. "
            "다시 만들려면 --rebuild 옵션을 사용하세요."
        )

        return vectorstore

    restaurants = load_restaurants()

    if limit is not None:
        restaurants = restaurants.head(
            limit
        )

    total = len(restaurants)

    print(
        f"임베딩 대상: {total:,}개 식당"
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        batch = restaurants.iloc[
            start:start + batch_size
        ]

        documents = [
            _row_to_document(row)
            for _, row in batch.iterrows()
        ]

        ids = (
            batch["RSTR_ID"]
            .astype("string")
            .tolist()
        )

        vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

        done = min(
            start + batch_size,
            total,
        )

        print(
            f"진행: {done:,}/{total:,}"
        )

    print(
        "ChromaDB 생성 완료: "
        f"{CHROMA_PERSIST_DIR}"
    )

    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    build_vectorstore(
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()