"""Разбор текста договора на пункты (clauses) и их нормализация."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# «1.», «1.2.», «1.2.3.», «2.1)», «10.4 » — нумерация пункта в начале абзаца.
NUMBER_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\s*[.)]?\s+(?=\S)")
# «Статья 5», «Раздел II», «ПРЕДМЕТ ДОГОВОРА» — заголовки разделов.
SECTION_RE = re.compile(
    r"^(?:(?:статья|раздел|глава)\s+[IVXLC\d]+|[\dIVXLC]+\s*[.)]\s+[А-ЯЁA-Z][^а-яёa-z]{3,})",
    re.IGNORECASE,
)
# Маркеры списков внутри пункта.
BULLET_RE = re.compile(r"^(?:[-–—•*]|[а-яa-z]\)|\d+\))\s+")

# Слова-«шум» для взвешивания схожести: встречаются почти в каждом пункте.
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    или ни быть был него до вас нибудь опять уж вам сказал ведь там потом себя ничего ей
    может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто человек
    чего раз тоже себе под жизнь будет ж тогда кто этот того потому этого какой совсем
    ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда зачем всех
    никогда сегодня можно при наконец два об другой хоть после над больше тот через эти
    нас про всего них какая много разве сказала три эту моя впрочем хорошо свою этой
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
    договор договора договору стороны сторона сторон сторонами настоящего настоящему
    настоящий настоящим пункт пункта пункте случае случаев соответствии течение
    """.split()
)


def normalize(text: str) -> str:
    """Нормализовать текст пункта для сравнения: регистр, пробелы, числа, кавычки."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[«»“”„\"']", " ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", " 0 ", text)  # числа → плейсхолдер
    text = re.sub(r"[^\w\s%]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    """Токены нормализованного текста (опционально без стоп-слов)."""
    words = normalize(text).split()
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return words


@dataclass
class Clause:
    """Один пункт договора."""

    index: int
    number: str | None
    section: str | None
    text: str
    source: str = ""
    normalized: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.normalized = normalize(self.text)

    @property
    def label(self) -> str:
        return f"п. {self.number}" if self.number else f"абз. {self.index + 1}"

    def short(self, limit: int = 220) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class Document:
    """Договор, разобранный на пункты."""

    name: str
    clauses: list[Clause]
    path: str = ""

    def __len__(self) -> int:
        return len(self.clauses)

    def by_number(self) -> dict[str, Clause]:
        return {c.number: c for c in self.clauses if c.number}


def _is_heading(text: str) -> bool:
    """Заголовок раздела: «Статья 5», «3. ПОРЯДОК ПРИЕМКИ», строка целиком в верхнем регистре."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters or len(text) > 160:
        return False
    if all(ch.isupper() for ch in letters):
        return True
    return bool(SECTION_RE.match(text)) and text.rstrip().endswith(tuple("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ"))


def split_clauses(paragraphs: list[str], source: str = "") -> list[Clause]:
    """Собрать абзацы в пункты: нумерованный абзац начинает пункт, остальное — продолжение."""
    clauses: list[Clause] = []
    section: str | None = None
    buffer: list[str] = []
    current_number: str | None = None

    def flush() -> None:
        nonlocal buffer, current_number
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                clauses.append(
                    Clause(
                        index=len(clauses),
                        number=current_number,
                        section=section,
                        text=text,
                        source=source,
                    )
                )
        buffer = []
        current_number = None

    for paragraph in paragraphs:
        if _is_heading(paragraph):
            flush()
            section = NUMBER_RE.sub("", paragraph).strip()
            continue

        match = NUMBER_RE.match(paragraph)
        if match:
            flush()
            current_number = match.group("num")
            buffer = [paragraph[match.end() :].strip()]
        elif BULLET_RE.match(paragraph) or buffer:
            buffer.append(BULLET_RE.sub("", paragraph).strip() if not buffer else paragraph.strip())
        else:
            buffer = [paragraph.strip()]

    flush()
    return [c for c in clauses if len(c.normalized) >= 15]


def parse_document(paragraphs: list[str], name: str, path: str = "") -> Document:
    return Document(name=name, clauses=split_clauses(paragraphs, source=name), path=path)
