from __future__ import annotations

import importlib
from collections.abc import Callable

GRAPH_MODULES = {
    "1": (
        "맛집 추천 챗봇",
        "restaurant_graph",
    ),
    "2": (
        "관광 계획 챗봇",
        "planner_graph",
    ),
}


def _load_runner(
    module_name: str,
) -> Callable[[str], dict]:
    try:
        module = importlib.import_module(
            module_name
        )

    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"{module_name}.py 파일을 찾을 수 없거나 "
            f"import에 실패했습니다: {error}"
        ) from error

    runner = getattr(
        module,
        "run",
        None,
    )

    if not callable(runner):
        raise RuntimeError(
            f"{module_name}.py에 "
            "run(question: str) -> dict "
            "함수가 필요합니다."
        )

    return runner


def main() -> None:
    print(
        "\n서울 여행 AI Graph 테스트"
    )

    for key, (
        label,
        _,
    ) in GRAPH_MODULES.items():
        print(f"{key}. {label}")

    print("0. 종료")

    while True:
        mode = input(
            "\n기능 선택: "
        ).strip()

        if mode == "0":
            break

        selected = GRAPH_MODULES.get(
            mode
        )

        if selected is None:
            print(
                "올바른 번호를 입력해주세요."
            )
            continue

        label, module_name = selected

        question = input(
            f"{label} 질문: "
        ).strip()

        if not question:
            print(
                "질문을 입력해주세요."
            )
            continue

        try:
            runner = _load_runner(
                module_name
            )

            result = runner(question)

            if isinstance(result, dict):
                answer = (
                    result.get("answer")
                    or result.get("error")
                )
            else:
                answer = str(result)

            print("\n결과")
            print(
                answer
                or "answer 값이 없습니다."
            )

        except Exception as error:
            print(
                f"\n실행 오류: {error}"
            )


if __name__ == "__main__":
    main()