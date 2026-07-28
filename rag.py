from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from data_loader import (
    search_restaurants_by_keyword,
)
from embedding import get_vectorstore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PERSIST_DIR = (
    BASE_DIR
    / os.getenv(
        "CHROMA_PERSIST_DIR",
        "chroma_db",
    )
)


def _vector_db_exists() -> bool:
    return (
        CHROMA_PERSIST_DIR
        / "chroma.sqlite3"
    ).exists()


def _document_to_result(
    document,
) -> dict:
    metadata = document.metadata

    return {
        "restaurant_id": str(
            metadata.get(
                "restaurant_id",
                "",
            )
        ),
        "name": metadata.get("name"),
        "address": metadata.get(
            "address"
        ),
        "latitude": metadata.get(
            "latitude"
        ),
        "longitude": metadata.get(
            "longitude"
        ),
        "landmark": metadata.get(
            "landmark"
        ),
        "landmark_distance": (
            metadata.get(
                "landmark_distance"
            )
        ),
        "business_hours": (
            metadata.get(
                "business_hours"
            )
        ),
        "parking_available": (
            metadata.get(
                "parking_available"
            )
        ),
        "pet_allowed": metadata.get(
            "pet_allowed"
        ),
        "foreign_menu_available": (
            metadata.get(
                "foreign_menu_available"
            )
        ),
        "content": document.page_content,
        "search_source": "vector",
    }


def search_restaurants(
    query: str,
    k: int = 10,
    use_vector: bool = True,
) -> list[dict]:
    query = query.strip()

    if not query or k <= 0:
        return []

    if (
        use_vector
        and _vector_db_exists()
    ):
        try:
            vectorstore = get_vectorstore()

            count = (
                vectorstore
                ._collection
                .count()
            )

            if count > 0:
                documents = (
                    vectorstore
                    .similarity_search(
                        query,
                        k=k,
                    )
                )

                return [
                    _document_to_result(
                        document
                    )
                    for document
                    in documents
                ]

        except Exception as error:
            warnings.warn(
                "벡터 검색에 실패하여 "
                "키워드 검색으로 전환합니다: "
                f"{error}",
                stacklevel=2,
            )

    results = (
        search_restaurants_by_keyword(
            query,
            limit=k,
        )
    )

    for result in results:
        result["search_source"] = (
            "keyword"
        )

    return results