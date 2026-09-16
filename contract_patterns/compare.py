"""Сопоставление договора института с исходным (эталонным) вариантом."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum

from .clauses import STOPWORDS, Clause, Document, normalize
from .similarity import Vectorizer, combined_similarity

MIN_EDIT_TOKENS = 2         # меньше значимых слов в правке — считаем её косметической
MATCH_THRESHOLD = 0.45      # ниже — у пункта нет пары в эталоне, значит он новый
WEAK_MATCH_THRESHOLD = 0.18  # порог для второго прохода: тот же номер/раздел, сильно переписан


class ChangeKind(str, Enum):
    ADDED = "added"          # института добавил пункт, которого нет в эталоне
    MODIFIED = "modified"    # пункт эталона переформулирован/дополнен
    REMOVED = "removed"      # пункт эталона выброшен
    UNCHANGED = "unchanged"


@dataclass
class Change:
    """Одно расхождение договора института с эталоном."""

    kind: ChangeKind
    institute: str
    clause: Clause | None                  # пункт в договоре института
    baseline_clause: Clause | None = None  # соответствующий пункт эталона
    similarity: float = 0.0
    inserted_text: str = ""                # что дописано относительно эталона
    deleted_text: str = ""                 # что убрано из эталона
    section: str | None = field(default=None)

    def __post_init__(self) -> None:
        if self.section is None:
            source = self.clause or self.baseline_clause
            self.section = source.section if source else None

    @property
    def payload(self) -> str:
        """Текст, по которому пункт сравнивается с правками других институтов."""
        if self.kind is ChangeKind.MODIFIED and self.inserted_text:
            return self.inserted_text
        if self.kind is ChangeKind.REMOVED and self.baseline_clause:
            return self.baseline_clause.text
        return self.clause.text if self.clause else ""


def edit_weight(text: str) -> int:
    """Сколько значимых слов в правке: числа считаются значимыми (срок 30 → 15 важен),
    служебные слова и падежные хвосты — нет."""
    words = normalize(text).split()
    return sum(1 for word in words if word not in STOPWORDS and (word == "0" or len(word) > 2))


def word_diff(baseline: str, revised: str) -> tuple[str, str]:
    """Вернуть (вставленное, удалённое) по словам между эталоном и правкой."""
    old = baseline.split()
    new = revised.split()
    matcher = difflib.SequenceMatcher(a=[w.lower() for w in old], b=[w.lower() for w in new])
    inserted: list[str] = []
    deleted: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            inserted.extend(new[j1:j2])
        if tag in ("delete", "replace"):
            deleted.extend(old[i1:i2])
    return " ".join(inserted), " ".join(deleted)


def _candidate_pairs(
    vectorizer: Vectorizer, baseline: Document, revision: Document
) -> list[tuple[float, Clause, Clause]]:
    baseline_by_number = baseline.by_number()
    pairs: list[tuple[float, Clause, Clause]] = []
    for clause in revision.clauses:
        seen: set[int] = set()
        candidates: list[Clause] = []
        if clause.number and clause.number in baseline_by_number:
            candidates.append(baseline_by_number[clause.number])
        candidates.extend(baseline.clauses)
        for candidate in candidates:
            if candidate.index in seen:
                continue
            seen.add(candidate.index)
            score = combined_similarity(vectorizer, candidate.text, clause.text)
            if candidate.number and candidate.number == clause.number:
                score = min(1.0, score + 0.05)  # совпадение номера — слабая подсказка
            if score >= MATCH_THRESHOLD:
                pairs.append((score, clause, candidate))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


def compare(baseline: Document, revision: Document, vectorizer: Vectorizer) -> list[Change]:
    """Сравнить договор института с эталоном и вернуть список расхождений."""
    matched_revision: dict[int, tuple[Clause, float]] = {}
    matched_baseline: set[int] = set()

    for score, clause, candidate in _candidate_pairs(vectorizer, baseline, revision):
        if clause.index in matched_revision or candidate.index in matched_baseline:
            continue
        matched_revision[clause.index] = (candidate, score)
        matched_baseline.add(candidate.index)

    # Второй проход: пункт переписан настолько, что похожесть низкая, но номер и раздел
    # совпадают — это правка эталонного пункта, а не новый пункт «из ниоткуда».
    leftovers = [c for c in baseline.clauses if c.index not in matched_baseline]
    for clause in revision.clauses:
        if clause.index in matched_revision or not clause.number:
            continue
        for candidate in leftovers:
            if candidate.index in matched_baseline or candidate.number != clause.number:
                continue
            score = combined_similarity(vectorizer, candidate.text, clause.text)
            if score >= WEAK_MATCH_THRESHOLD:
                matched_revision[clause.index] = (candidate, score)
                matched_baseline.add(candidate.index)
            break

    changes: list[Change] = []
    for clause in revision.clauses:
        match = matched_revision.get(clause.index)
        if match is None:
            changes.append(
                Change(kind=ChangeKind.ADDED, institute=revision.name, clause=clause)
            )
            continue
        baseline_clause, score = match
        inserted, deleted = word_diff(baseline_clause.text, clause.text)
        if edit_weight(inserted) < MIN_EDIT_TOKENS and edit_weight(deleted) < MIN_EDIT_TOKENS:
            # правка чисто косметическая (пробелы, падежи, номера)
            changes.append(
                Change(
                    kind=ChangeKind.UNCHANGED,
                    institute=revision.name,
                    clause=clause,
                    baseline_clause=baseline_clause,
                    similarity=score,
                )
            )
            continue
        changes.append(
            Change(
                kind=ChangeKind.MODIFIED,
                institute=revision.name,
                clause=clause,
                baseline_clause=baseline_clause,
                similarity=score,
                inserted_text=inserted,
                deleted_text=deleted,
            )
        )

    for clause in baseline.clauses:
        if clause.index not in matched_baseline:
            changes.append(
                Change(
                    kind=ChangeKind.REMOVED,
                    institute=revision.name,
                    clause=None,
                    baseline_clause=clause,
                )
            )

    return changes
