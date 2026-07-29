from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

GRAPH_MODULES = {
    "1": {
        "key": "restaurant",
        "label": "맛집 추천 챗봇",
        "module": "restaurant.restaurant_graph",
    },
    "2": {
        "key": "planner",
        "label": "관광 계획 챗봇",
        "module": "planner.planner_graph",
    },
}

MODE_ALIASES = {
    "1": "1",
    "restaurant": "1",
    "맛집": "1",
    "2": "2",
    "planner": "2",
    "관광": "2",
}

Runner = Callable[..., dict[str, Any]]


def _load_runner(module_name: str) -> Runner:
    """도메인 Graph 모듈에서 공통 run 함수를 불러온다."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or "알 수 없는 모듈"
        raise RuntimeError(
            f"{module_name} import에 실패했습니다. "
            f"누락된 모듈: {missing_name}. "
            "가상환경을 활성화하고 pip install -r requirements.txt를 실행해주세요."
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"{module_name} 초기화에 실패했습니다: {error}"
        ) from error

    runner = getattr(module, "run", None)

    if not callable(runner):
        raise RuntimeError(
            f"{module_name}에 run(question: str, previous_state=None) "
            "함수가 필요합니다."
        )

    return runner


def _invoke_runner(
    runner: Runner,
    question: str,
    previous_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """이전 State를 전달해 후속 질문과 일정 수정 요청을 지원한다."""
    try:
        result = runner(
            question,
            previous_state=previous_state,
        )
    except TypeError as error:
        if "previous_state" not in str(error):
            raise
        result = runner(question)

    if not isinstance(result, dict):
        raise TypeError(
            "Graph run 함수는 dict 형태의 State를 반환해야 합니다."
        )

    return result


def _print_result(result: dict[str, Any]) -> None:
    """Graph의 최종 답변과 점검 정보를 콘솔에 표시한다."""
    answer = result.get("answer")
    error = result.get("error")

    print("\n결과")
    print(answer or error or "answer 값이 없습니다.")

    status = result.get("search_status")
    if status:
        print(f"\n검색 상태: {status}")

    warnings_list = result.get("warnings") or []
    if warnings_list:
        print("\n경고 및 확인 필요")
        for warning in warnings_list:
            print(f"- {warning}")

    unsupported = result.get("unsupported_conditions") or []
    if unsupported:
        print("\n현재 지원하지 않는 조건")
        for item in unsupported:
            print(f"- {item}")

    if error and answer:
        print(f"\n내부 오류 기록: {error}")


def run_health_check() -> bool:
    """LLM 호출 없이 데이터·ChromaDB·환경변수 연결 상태를 점검한다."""
    success = True
    print("서울 여행 AI 실행 환경 점검\n")

    try:
        from data_loader import get_data_info

        data_info = get_data_info()
        print("[데이터]")
        print(json.dumps(data_info, ensure_ascii=False, indent=2))
    except Exception as error:
        success = False
        print("[데이터] 실패")
        print(f"- {error}")

    try:
        from embedding import get_vectorstore_info

        vector_info = get_vectorstore_info()
        print("\n[ChromaDB]")
        print(json.dumps(vector_info, ensure_ascii=False, indent=2))
    except Exception as error:
        success = False
        print("\n[ChromaDB] 실패")
        print(f"- {error}")

    try:
        from llm_loader import get_llm_settings

        llm_info = get_llm_settings()
        print("\n[LLM 설정]")
        print(json.dumps(llm_info, ensure_ascii=False, indent=2))

        if not llm_info.get("api_key_configured"):
            success = False
            print("- OPENAI_API_KEY가 설정되지 않았습니다.")

        if not llm_info.get("model"):
            success = False
            print("- LLM_AI_MODEL이 설정되지 않았습니다.")
    except Exception as error:
        success = False
        print("\n[LLM 설정] 실패")
        print(f"- {error}")

    print("\n점검 결과:", "통과" if success else "확인 필요")
    return success


def run_once(mode: str, question: str) -> dict[str, Any]:
    """명령행 인자로 Graph 한 번을 실행한다."""
    mode_key = MODE_ALIASES.get(mode.strip().casefold())

    if mode_key is None:
        raise ValueError(
            "mode는 restaurant 또는 planner여야 합니다."
        )

    selected = GRAPH_MODULES[mode_key]
    runner = _load_runner(str(selected["module"]))
    return _invoke_runner(runner, question.strip(), None)


def interactive_main() -> None:
    """두 Graph를 번갈아 시험할 수 있는 콘솔 테스트 화면."""
    print("\n서울 여행 AI Graph 테스트")
    print("같은 모드에서 이전 결과를 유지하므로 후속 질문을 시험할 수 있습니다.")

    states: dict[str, dict[str, Any] | None] = {
        "restaurant": None,
        "planner": None,
    }

    while True:
        print("\n1. 맛집 추천 챗봇")
        print("2. 관광 계획 챗봇")
        print("c. 데이터·ChromaDB·LLM 설정 점검")
        print("r. 이전 대화 상태 초기화")
        print("0. 종료")

        mode = input("\n기능 선택: ").strip().casefold()

        if mode == "0":
            break

        if mode == "c":
            run_health_check()
            continue

        if mode == "r":
            states = {
                "restaurant": None,
                "planner": None,
            }
            print("이전 대화 상태를 초기화했습니다.")
            continue

        selected = GRAPH_MODULES.get(mode)

        if selected is None:
            print("올바른 번호를 입력해주세요.")
            continue

        question = input(
            f"{selected['label']} 질문: "
        ).strip()

        if not question:
            print("질문을 입력해주세요.")
            continue

        domain_key = str(selected["key"])

        try:
            runner = _load_runner(str(selected["module"]))
            result = _invoke_runner(
                runner,
                question,
                states[domain_key],
            )
            states[domain_key] = result
            _print_result(result)
        except KeyboardInterrupt:
            print("\n실행을 중단했습니다.")
        except Exception as error:
            print(f"\n실행 오류: {error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="서울 여행 AI Graph 콘솔 테스트"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="LLM 호출 없이 데이터와 설정 상태를 점검합니다.",
    )
    parser.add_argument(
        "--mode",
        choices=["restaurant", "planner"],
        help="한 번만 실행할 Graph를 선택합니다.",
    )
    parser.add_argument(
        "--question",
        help="--mode와 함께 전달할 사용자 질문입니다.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.check:
        passed = run_health_check()
        raise SystemExit(0 if passed else 1)

    if args.mode or args.question:
        if not args.mode or not args.question:
            parser.error("--mode와 --question은 함께 사용해야 합니다.")

        try:
            result = run_once(args.mode, args.question)
            _print_result(result)
        except Exception as error:
            print(f"실행 오류: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        return

    interactive_main()


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
