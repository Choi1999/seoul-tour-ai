from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    model_name = os.getenv(
        "LLM_AI_MODEL"
    )

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
        temperature=float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0",
            )
        ),
    )