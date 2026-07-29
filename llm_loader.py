from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name}은 숫자여야 합니다: {value}"
        ) from error


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name}은 정수여야 합니다: {value}"
        ) from error


def get_llm_settings() -> dict[str, Any]:
    """API 키 원문을 노출하지 않고 현재 LLM 설정 상태를 반환한다."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = (
        os.getenv("LLM_AI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or ""
    ).strip()

    return {
        "api_key_configured": bool(api_key),
        "model": model_name,
        "temperature": _read_float("LLM_TEMPERATURE", 0.0),
        "timeout_seconds": _read_float("LLM_TIMEOUT_SECONDS", 60.0),
        "max_retries": _read_int("LLM_MAX_RETRIES", 2),
    }


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Restaurant와 Planner가 함께 사용하는 ChatOpenAI 객체를 한 번만 생성한다."""
    settings = get_llm_settings()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = str(settings["model"]).strip()

    if not api_key:
        raise RuntimeError(
            ".env에 OPENAI_API_KEY를 설정해주세요."
        )

    if not model_name:
        raise RuntimeError(
            ".env에 LLM_AI_MODEL을 설정해주세요."
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=float(settings["temperature"]),
        timeout=float(settings["timeout_seconds"]),
        max_retries=int(settings["max_retries"]),
    )


def clear_llm_cache() -> None:
    """개발 중 환경변수를 바꾼 뒤 LLM 객체를 다시 생성할 때 사용한다."""
    get_llm.cache_clear()
