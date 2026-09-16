"""CLI агента-помощника по типовому договору."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import analyze
from .extract import ExtractionError
from .report import to_json, to_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contract-patterns",
        description=(
            "Агент-помощник: сравнивает договоры институтов с исходным вариантом "
            "на согласование, находит общие добавленные паттерны и предлагает, "
            "как поправить типовой договор."
        ),
    )
    parser.add_argument("--baseline", "-b", required=True, help="исходный вариант на согласование")
    parser.add_argument(
        "institutes",
        nargs="+",
        help="договоры институтов (.docx / .txt / .pdf), можно указать каталог",
    )
    parser.add_argument("--out", "-o", help="файл отчёта Markdown (по умолчанию — stdout)")
    parser.add_argument("--json", dest="json_out", help="файл отчёта JSON")
    parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="сколько институтов должны внести правку, чтобы она считалась общей",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="дополнить рекомендации формулировками Claude (нужен ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--model", default=None, help="модель Claude для --llm")
    return parser


def _expand(paths: list[str], exclude: set[Path]) -> list[Path]:
    """Развернуть каталоги в списки файлов договоров, исключив эталон и файлы отчёта."""
    result: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            result.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.suffix.lower() in (".docx", ".txt", ".md", ".pdf", ".doc")
                )
            )
        else:
            result.append(path)
    return [path for path in result if path.resolve() not in exclude]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude = {
        Path(value).resolve()
        for value in (args.baseline, args.out, args.json_out)
        if value
    }
    institute_paths = _expand(args.institutes, exclude)
    if not institute_paths:
        print("Не найдено ни одного договора института", file=sys.stderr)
        return 2

    try:
        result = analyze(args.baseline, list(institute_paths), min_support=args.min_support)
    except (ExtractionError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if args.llm and result.recommendations:
        from .llm import DEFAULT_MODEL, enrich

        try:
            enrich(result.recommendations, model=args.model or DEFAULT_MODEL)
        except RuntimeError as exc:
            print(f"Предупреждение: слой Claude отключён — {exc}", file=sys.stderr)

    markdown = to_markdown(
        result.baseline, result.revisions, result.changes,
        result.recommendations, result.min_support,
    )
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"Отчёт сохранён: {args.out}", file=sys.stderr)
    else:
        print(markdown)

    if args.json_out:
        Path(args.json_out).write_text(
            to_json(
                result.baseline, result.revisions, result.changes,
                result.recommendations, result.min_support,
            ),
            encoding="utf-8",
        )
        print(f"JSON сохранён: {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
