from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


CHROMA_PATH = _resolve_project_path(
    os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
)
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "langchain"
).strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-m3"
).strip()
EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE", "cpu"
).strip()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """기존 DB 조회에 사용할 임베딩 모델만 로드한다."""
    model_kwargs: dict[str, Any] = {}
    if EMBEDDING_DEVICE:
        model_kwargs["device"] = EMBEDDING_DEVICE

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
    )


@lru_cache(maxsize=1)
def get_chroma_client() -> Any:
    """기존 ChromaDB를 열고 컬렉션 존재 여부를 확인한다."""
    sqlite_path = CHROMA_PATH / "chroma.sqlite3"

    if not CHROMA_PATH.is_dir():
        raise FileNotFoundError(f"ChromaDB 폴더가 없습니다: {CHROMA_PATH}")
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"ChromaDB 파일이 없습니다: {sqlite_path}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as error:
        try:
            collection_names = [
                getattr(collection, "name", str(collection))
                for collection in client.list_collections()
            ]
        except Exception:
            collection_names = []

        raise RuntimeError(
            "기존 ChromaDB 컬렉션 연결에 실패했습니다. "
            f"요청 컬렉션={CHROMA_COLLECTION_NAME}, "
            f"확인 컬렉션={collection_names}"
        ) from error

    return client


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """문서 추가나 재임베딩 없이 기존 컬렉션을 조회용으로 반환한다."""
    return Chroma(
        client=get_chroma_client(),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def get_vectorstore_info() -> dict[str, Any]:
    collection = get_chroma_client().get_collection(
        name=CHROMA_COLLECTION_NAME
    )
    return {
        "chroma_path": str(CHROMA_PATH),
        "collection_name": CHROMA_COLLECTION_NAME,
        "document_count": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": EMBEDDING_DEVICE,
    }


def clear_embedding_cache() -> None:
    get_vectorstore.cache_clear()
    get_chroma_client.cache_clear()
    get_embeddings.cache_clear()
