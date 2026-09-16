"""Группировка правок разных институтов в общие паттерны."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .clauses import Clause, tokens
from .compare import Change, ChangeKind
from .similarity import Vectorizer, combined_similarity, shingles
from .topics import topic_of

CLUSTER_THRESHOLD = 0.42   # схожесть правок, при которой они считаются одним паттерном
TOPIC_BONUS = 0.15         # надбавка, если правки относятся к одной теме договора


@dataclass
class Pattern:
    """Правка, которую независимо внесли несколько институтов."""

    kind: ChangeKind
    changes: list[Change] = field(default_factory=list)
    baseline_clause: Clause | None = None

    @property
    def institutes(self) -> list[str]:
        seen: dict[str, None] = {}
        for change in self.changes:
            seen.setdefault(change.institute, None)
        return list(seen)

    @property
    def support(self) -> int:
        """Сколько институтов независимо внесли эту правку."""
        return len(self.institutes)

    @property
    def section(self) -> str | None:
        sections = Counter(c.section for c in self.changes if c.section)
        return sections.most_common(1)[0][0] if sections else None

    @property
    def representative(self) -> Change:
        """Самая «средняя» формулировка паттерна — её показываем человеку."""
        texts = [c.payload for c in self.changes]
        if len(self.changes) == 1:
            return self.changes[0]
        vectorizer = Vectorizer(texts)
        best, best_score = self.changes[0], -1.0
        for change in self.changes:
            score = sum(
                combined_similarity(vectorizer, change.payload, other.payload)
                for other in self.changes
                if other is not change
            )
            if score > best_score:
                best, best_score = change, score
        return best

    def common_phrases(self, min_words: int = 4, limit: int = 5) -> list[str]:
        """Формулировки, дословно повторяющиеся минимум у половины институтов."""
        per_institute: dict[str, set[tuple[str, ...]]] = {}
        for change in self.changes:
            per_institute.setdefault(change.institute, set()).update(
                shingles(change.payload, size=min_words)
            )
        if len(per_institute) < 2:
            return []
        counts: Counter[tuple[str, ...]] = Counter()
        for grams in per_institute.values():
            counts.update(grams)
        needed = max(2, (len(per_institute) + 1) // 2)
        frequent = [gram for gram, count in counts.items() if count >= needed]
        merged = [gram for gram in _merge_grams(frequent) if _informative(gram, min_words)]
        merged.sort(key=len, reverse=True)
        return [" ".join(gram) for gram in merged[:limit]]

    def keywords(self, limit: int = 8) -> list[str]:
        counts: Counter[str] = Counter()
        for change in self.changes:
            counts.update(set(tokens(change.payload)))
        return [word for word, _ in counts.most_common(limit)]


def _informative(gram: tuple[str, ...], min_words: int) -> bool:
    """Отсечь фразы из плейсхолдеров чисел и обрывки короче исходной n-граммы."""
    if len(gram) < min_words:
        return False
    placeholders = sum(1 for word in gram if word == "0")
    return placeholders <= len(gram) // 3


def _merge_grams(grams: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Склеить пересекающиеся n-граммы в более длинные фразы."""
    merged = list(grams)
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(merged):
            for j, right in enumerate(merged):
                if i == j:
                    continue
                overlap = min(len(left), len(right)) - 1
                while overlap > 0:
                    if left[-overlap:] == right[:overlap]:
                        merged[i] = left + right[overlap:]
                        merged.pop(j)
                        changed = True
                        break
                    overlap -= 1
                if changed:
                    break
            if changed:
                break
    return merged


def _cluster_key(change: Change) -> str:
    return "removed" if change.kind is ChangeKind.REMOVED else "edit"


def build_patterns(changes: list[Change], min_support: int = 2) -> list[Pattern]:
    """Сгруппировать правки в паттерны и оставить те, что встречаются у min_support институтов."""
    relevant = [c for c in changes if c.kind is not ChangeKind.UNCHANGED and c.payload]
    if not relevant:
        return []

    vectorizer = Vectorizer([c.payload for c in relevant])
    clusters: list[Pattern] = []

    # Удалённые пункты группируем строго по пункту эталона — там сходство очевидно.
    removed: dict[int, Pattern] = {}
    for change in relevant:
        if change.kind is not ChangeKind.REMOVED or change.baseline_clause is None:
            continue
        pattern = removed.setdefault(
            change.baseline_clause.index,
            Pattern(kind=ChangeKind.REMOVED, baseline_clause=change.baseline_clause),
        )
        pattern.changes.append(change)
    clusters.extend(removed.values())

    edits = [c for c in relevant if _cluster_key(c) == "edit"]
    edits.sort(key=lambda c: len(tokens(c.payload)), reverse=True)
    edit_clusters: list[Pattern] = []
    for change in edits:
        best_cluster, best_score = None, 0.0
        change_topic = topic_of(change.payload)
        for cluster in edit_clusters:
            score = max(
                combined_similarity(vectorizer, change.payload, member.payload)
                + (
                    TOPIC_BONUS
                    if change_topic is not None and change_topic == topic_of(member.payload)
                    else 0.0
                )
                for member in cluster.changes
            )
            if score > best_score:
                best_cluster, best_score = cluster, score
        if best_cluster is not None and best_score >= CLUSTER_THRESHOLD:
            best_cluster.changes.append(change)
            if best_cluster.baseline_clause is None and change.baseline_clause is not None:
                best_cluster.baseline_clause = change.baseline_clause
        else:
            edit_clusters.append(
                Pattern(
                    kind=change.kind,
                    changes=[change],
                    baseline_clause=change.baseline_clause,
                )
            )
    clusters.extend(edit_clusters)

    for cluster in clusters:
        kinds = Counter(c.kind for c in cluster.changes)
        cluster.kind = kinds.most_common(1)[0][0]

    result = [c for c in clusters if c.support >= min_support]
    result.sort(key=lambda p: (p.support, len(p.changes)), reverse=True)
    return result
