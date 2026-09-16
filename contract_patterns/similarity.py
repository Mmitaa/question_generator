"""Метрики схожести текстов пунктов (чистый stdlib, без numpy/sklearn)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from .clauses import tokens


class Vectorizer:
    """TF-IDF по корпусу пунктов: редкие формулировки весят больше шаблонных."""

    def __init__(self, corpus: Sequence[str]) -> None:
        self.n_docs = max(len(corpus), 1)
        document_frequency: Counter[str] = Counter()
        for text in corpus:
            document_frequency.update(set(tokens(text)))
        self.idf = {
            term: math.log((self.n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in document_frequency.items()
        }
        self._default_idf = math.log(self.n_docs + 1) + 1.0

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokens(text))
        if not counts:
            return {}
        total = sum(counts.values())
        vec = {
            term: (count / total) * self.idf.get(term, self._default_idf)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vec.values()))
        return {term: value / norm for term, value in vec.items()} if norm else {}

    def similarity(self, left: str, right: str) -> float:
        return cosine(self.vector(left), self.vector(right))


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Словесные n-граммы — для устойчивого сравнения формулировок."""
    words = tokens(text, drop_stopwords=False)
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: Iterable, right: Iterable) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def combined_similarity(vectorizer: Vectorizer, left: str, right: str) -> float:
    """Косинус TF-IDF плюс перекрытие 3-грамм: смысл + дословность формулировки."""
    return 0.65 * vectorizer.similarity(left, right) + 0.35 * jaccard(
        shingles(left), shingles(right)
    )
