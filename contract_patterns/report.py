"""Формирование отчёта: Markdown для человека и JSON для дальнейшей обработки."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

from .clauses import Document
from .compare import Change, ChangeKind
from .recommend import Recommendation

KIND_LABEL = {
    ChangeKind.ADDED: "новый пункт",
    ChangeKind.MODIFIED: "переформулирован пункт",
    ChangeKind.REMOVED: "пункт исключён",
    ChangeKind.UNCHANGED: "без изменений",
}


def _stats(changes: list[Change]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for change in changes:
        row = stats.setdefault(
            change.institute, {kind.value: 0 for kind in ChangeKind}
        )
        row[change.kind.value] += 1
    return stats


def to_markdown(
    baseline: Document,
    revisions: list[Document],
    changes: list[Change],
    recommendations: list[Recommendation],
    min_support: int,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Анализ правок институтов в типовом договоре")
    add("")
    add(f"_Дата отчёта: {date.today().isoformat()}_")
    add("")
    add(f"- Эталон: **{baseline.name}** ({len(baseline)} пунктов)")
    add(f"- Договоров институтов: **{len(revisions)}**")
    add(f"- Порог общности паттерна: правка встречается минимум у **{min_support}** институтов")
    add(f"- Найдено общих паттернов: **{len(recommendations)}**")
    add("")

    add("## Сводка по институтам")
    add("")
    add("| Институт | Пунктов | Новых | Изменённых | Исключённых |")
    add("| --- | ---: | ---: | ---: | ---: |")
    stats = _stats(changes)
    for revision in revisions:
        row = stats.get(revision.name, {})
        add(
            f"| {revision.name} | {len(revision)} | {row.get('added', 0)} | "
            f"{row.get('modified', 0)} | {row.get('removed', 0)} |"
        )
    add("")

    if not recommendations:
        add("## Рекомендации")
        add("")
        add(
            "Общих паттернов не найдено: правки институтов не совпадают между собой "
            "либо порог `--min-support` слишком высок. Попробуйте понизить порог."
        )
        return "\n".join(lines)

    add("## Что предлагается изменить в эталоне")
    add("")
    add("| # | Тема | Действие | Институтов | Приоритет |")
    add("| ---: | --- | --- | ---: | --- |")
    for number, rec in enumerate(recommendations, start=1):
        add(
            f"| {number} | {rec.title} | {rec.action} | "
            f"{rec.support}/{rec.total_institutes} | {rec.priority} |"
        )
    add("")

    add("## Подробно по каждому паттерну")
    add("")
    for number, rec in enumerate(recommendations, start=1):
        add(f"### {number}. {rec.title}")
        add("")
        add(f"- **Действие:** {rec.action}")
        add(f"- **Место в договоре:** {rec.target}")
        if rec.section:
            add(f"- **Раздел:** {rec.section}")
        add(
            f"- **Распространённость:** {rec.support} из {rec.total_institutes} "
            f"({', '.join(rec.institutes)}) — приоритет {rec.priority}"
        )
        add(f"- **Почему:** {rec.rationale}")
        add("")
        if rec.common_phrases:
            add("**Формулировки, совпадающие дословно у нескольких институтов:**")
            add("")
            for phrase in rec.common_phrases:
                add(f"- «…{phrase}…»")
            add("")
        if rec.llm_summary:
            add("**Комментарий помощника:**")
            add("")
            for paragraph in rec.llm_summary.split("\n\n"):
                add(paragraph.strip())
                add("")
        wording = rec.llm_wording or rec.proposed_text
        if wording:
            add("**Предлагаемая редакция для эталона:**")
            add("")
            add("> " + wording.replace("\n", "\n> "))
            add("")
        if rec.examples:
            add("<details><summary>Как это написали институты</summary>")
            add("")
            for institute, text in rec.examples:
                add(f"- **{institute}:** {text}")
            add("")
            add("</details>")
            add("")

    add("---")
    add("")
    add(
        "Отчёт подготовлен автоматически. Перед внесением правок в типовой договор "
        "решение принимает юрист: агент показывает статистику и черновые формулировки, "
        "а не согласовывает редакцию."
    )
    return "\n".join(lines)


def to_json(
    baseline: Document,
    revisions: list[Document],
    changes: list[Change],
    recommendations: list[Recommendation],
    min_support: int,
) -> str:
    payload = {
        "baseline": {"name": baseline.name, "path": baseline.path, "clauses": len(baseline)},
        "institutes": [
            {"name": r.name, "path": r.path, "clauses": len(r)} for r in revisions
        ],
        "min_support": min_support,
        "stats": _stats(changes),
        "recommendations": [
            {
                **{k: v for k, v in asdict(rec).items() if k != "kind"},
                "kind": rec.kind.value,
                "priority": rec.priority,
                "coverage": round(rec.coverage, 3),
            }
            for rec in recommendations
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
