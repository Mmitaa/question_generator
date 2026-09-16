"""Превращение общих паттернов в рекомендации по правке эталонного договора."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cluster import Pattern
from .compare import ChangeKind
from .topics import TOPICS, topic_of

ACTION_BY_KIND = {
    ChangeKind.ADDED: "добавить в эталон новый пункт",
    ChangeKind.MODIFIED: "переформулировать существующий пункт эталона",
    ChangeKind.REMOVED: "пересмотреть пункт эталона (институты его исключают)",
}


@dataclass
class Recommendation:
    """Предложение человеку, как поправить исходный договор на согласование."""

    title: str
    action: str
    support: int
    total_institutes: int
    institutes: list[str]
    target: str
    section: str | None
    proposed_text: str
    rationale: str
    examples: list[tuple[str, str]] = field(default_factory=list)
    common_phrases: list[str] = field(default_factory=list)
    kind: ChangeKind = ChangeKind.ADDED
    llm_summary: str = ""
    llm_wording: str = ""

    @property
    def coverage(self) -> float:
        return self.support / self.total_institutes if self.total_institutes else 0.0

    @property
    def priority(self) -> str:
        if self.coverage >= 0.8:
            return "высокий"
        if self.coverage >= 0.5:
            return "средний"
        return "низкий"


def detect_topic(pattern: Pattern) -> str:
    text = " ".join(change.payload for change in pattern.changes)
    topic = topic_of(text)
    if topic is None and pattern.baseline_clause is not None:
        # у короткой правки («но не более 10% от цены Договора») тему задаёт сам пункт эталона
        topic = topic_of(pattern.baseline_clause.text)
    if topic:
        return topic
    keywords = pattern.keywords(limit=4)
    return "Правка: " + ", ".join(keywords) if keywords else "Правка без явной темы"


def build_recommendation(pattern: Pattern, total_institutes: int) -> Recommendation:
    representative = pattern.representative
    if pattern.baseline_clause is not None:
        target = f"{pattern.baseline_clause.label}: {pattern.baseline_clause.short(140)}"
    elif pattern.section:
        target = f"раздел «{pattern.section}» (новый пункт)"
    else:
        target = "новый пункт (раздел определить вручную)"

    proposed = (
        representative.clause.text
        if representative.clause is not None
        else (representative.baseline_clause.text if representative.baseline_clause else "")
    )

    rationale = (
        f"{pattern.support} из {total_institutes} институтов "
        f"({', '.join(pattern.institutes)}) внесли эту правку независимо друг от друга. "
    )
    if pattern.kind is ChangeKind.REMOVED:
        rationale += "Пункт эталона у них отсутствует — вероятно, он неприемлем или избыточен."
    elif pattern.kind is ChangeKind.MODIFIED:
        rationale += "Формулировка эталона переписывается каждый раз — её стоит закрепить заранее."
    else:
        rationale += "Пункта в эталоне нет, институты дописывают его сами."

    examples = [
        (change.institute, (change.clause or change.baseline_clause).short(300))  # type: ignore[union-attr]
        for change in pattern.changes
        if change.clause or change.baseline_clause
    ][:4]

    return Recommendation(
        title=detect_topic(pattern),
        action=ACTION_BY_KIND[pattern.kind],
        support=pattern.support,
        total_institutes=total_institutes,
        institutes=pattern.institutes,
        target=target,
        section=pattern.section,
        proposed_text=proposed,
        rationale=rationale,
        examples=examples,
        common_phrases=pattern.common_phrases(),
        kind=pattern.kind,
    )


def build_recommendations(patterns: list[Pattern], total_institutes: int) -> list[Recommendation]:
    recommendations = [build_recommendation(p, total_institutes) for p in patterns]
    recommendations.sort(key=lambda r: (r.support, len(r.common_phrases)), reverse=True)
    return recommendations
