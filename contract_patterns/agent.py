"""Основной пайплайн агента: эталон + договоры институтов → рекомендации."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .clauses import Document, parse_document
from .cluster import Pattern, build_patterns
from .compare import Change, compare
from .extract import extract_paragraphs
from .recommend import Recommendation, build_recommendations
from .similarity import Vectorizer


@dataclass
class AnalysisResult:
    baseline: Document
    revisions: list[Document]
    changes: list[Change]
    patterns: list[Pattern]
    recommendations: list[Recommendation]
    min_support: int


def load_document(path: str | Path, name: str | None = None) -> Document:
    path = Path(path)
    paragraphs = extract_paragraphs(path)
    return parse_document(paragraphs, name=name or path.stem, path=str(path))


def analyze(
    baseline_path: str | Path,
    institute_paths: list[str | Path],
    min_support: int | None = None,
) -> AnalysisResult:
    """Проанализировать договоры институтов относительно эталона."""
    baseline = load_document(baseline_path)
    revisions = [load_document(path) for path in institute_paths]
    if not revisions:
        raise ValueError("Нужен хотя бы один договор института")

    if min_support is None:
        # по умолчанию: правка считается общей, если её сделали минимум два института
        # и не меньше трети от их общего числа
        min_support = max(2, round(len(revisions) / 3))
    min_support = min(min_support, len(revisions))

    corpus = [c.text for c in baseline.clauses]
    for revision in revisions:
        corpus.extend(c.text for c in revision.clauses)
    vectorizer = Vectorizer(corpus)

    changes: list[Change] = []
    for revision in revisions:
        changes.extend(compare(baseline, revision, vectorizer))

    patterns = build_patterns(changes, min_support=min_support)
    recommendations = build_recommendations(patterns, total_institutes=len(revisions))

    return AnalysisResult(
        baseline=baseline,
        revisions=revisions,
        changes=changes,
        patterns=patterns,
        recommendations=recommendations,
        min_support=min_support,
    )
