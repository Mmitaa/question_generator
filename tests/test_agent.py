"""Тесты агента: разбор, сравнение, кластеризация, сквозной прогон."""

from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from contract_patterns.agent import analyze
from contract_patterns.clauses import parse_document
from contract_patterns.cluster import build_patterns
from contract_patterns.compare import ChangeKind, compare, word_diff
from contract_patterns.extract import extract_paragraphs
from contract_patterns.report import to_json, to_markdown
from contract_patterns.similarity import Vectorizer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
INSTITUTES = [EXAMPLES / "institute_a.txt", EXAMPLES / "institute_b.txt", EXAMPLES / "institute_v.txt"]

DOCX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>1. ПРЕДМЕТ ДОГОВОРА</w:t></w:r></w:p>
<w:p><w:r><w:t>1.1. Исполнитель обязуется оказать услуги </w:t></w:r><w:r><w:t>надлежащего качества.</w:t></w:r></w:p>
</w:body></w:document>"""


class ExtractTest(unittest.TestCase):
    def test_docx_paragraphs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.docx"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("word/document.xml", DOCX_XML)
            paragraphs = extract_paragraphs(path)
        self.assertEqual(paragraphs[0], "1. ПРЕДМЕТ ДОГОВОРА")
        self.assertEqual(paragraphs[1], "1.1. Исполнитель обязуется оказать услуги надлежащего качества.")


class ClauseTest(unittest.TestCase):
    def test_numbering_and_sections(self) -> None:
        document = parse_document(
            [
                "2. ЦЕНА ДОГОВОРА",
                "2.1. Цена Договора составляет 100 рублей, в том числе НДС.",
                "2.2. Оплата производится в течение 30 календарных дней с даты подписания Акта.",
                "Продолжение пункта об оплате и порядке расчетов между Сторонами.",
            ],
            name="эталон",
        )
        self.assertEqual([c.number for c in document.clauses], ["2.1", "2.2"])
        self.assertEqual(document.clauses[0].section, "ЦЕНА ДОГОВОРА")
        self.assertIn("Продолжение пункта", document.clauses[1].text)

    def test_word_diff(self) -> None:
        inserted, deleted = word_diff(
            "Оплата производится в течение 30 календарных дней.",
            "Оплата производится в течение 15 рабочих дней после получения счета.",
        )
        self.assertIn("рабочих", inserted)
        self.assertIn("календарных", deleted)


class CompareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = parse_document(
            [
                "1.1. Исполнитель обязуется оказать Заказчику услуги согласно Техническому заданию.",
                "1.2. Заказчик оплачивает услуги в течение 30 календарных дней с даты подписания Акта.",
                "1.3. Споры рассматриваются в арбитражном суде по месту нахождения Заказчика.",
            ],
            name="эталон",
        )
        self.revision = parse_document(
            [
                "1.1. Исполнитель обязуется оказать Заказчику услуги согласно Техническому заданию.",
                "1.2. Заказчик оплачивает услуги в течение 15 рабочих дней с даты подписания Акта.",
                "1.4. Стороны обязуются соблюдать требования антикоррупционного законодательства.",
            ],
            name="институт",
        )
        corpus = [c.text for c in self.baseline.clauses] + [c.text for c in self.revision.clauses]
        self.changes = compare(self.baseline, self.revision, Vectorizer(corpus))

    def _kinds(self, kind: ChangeKind) -> list:
        return [c for c in self.changes if c.kind is kind]

    def test_unchanged_modified_added_removed(self) -> None:
        self.assertEqual(len(self._kinds(ChangeKind.UNCHANGED)), 1)
        modified = self._kinds(ChangeKind.MODIFIED)
        self.assertEqual(len(modified), 1)
        self.assertIn("рабочих", modified[0].inserted_text)
        self.assertEqual(len(self._kinds(ChangeKind.ADDED)), 1)
        removed = self._kinds(ChangeKind.REMOVED)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].baseline_clause.number, "1.3")


class PatternTest(unittest.TestCase):
    def test_min_support_filters_single_institute_edits(self) -> None:
        result = analyze(EXAMPLES / "baseline.txt", list(INSTITUTES), min_support=2)
        self.assertTrue(result.patterns)
        self.assertTrue(all(p.support >= 2 for p in result.patterns))

        strict = build_patterns(result.changes, min_support=4)
        self.assertEqual(strict, [])

    def test_common_patterns_are_found(self) -> None:
        result = analyze(EXAMPLES / "baseline.txt", list(INSTITUTES), min_support=2)
        titles = {rec.title for rec in result.recommendations}
        for expected in (
            "Антикоррупционная оговорка",
            "Конфиденциальность",
            "Персональные данные",
            "Электронный документооборот и подписи",
            "Ответственность и неустойка",
        ):
            self.assertIn(expected, titles)

        anticorruption = next(r for r in result.recommendations if r.title == "Антикоррупционная оговорка")
        self.assertEqual(anticorruption.support, 3)
        self.assertEqual(anticorruption.priority, "высокий")
        self.assertTrue(anticorruption.common_phrases)

    def test_default_min_support_scales_with_input(self) -> None:
        result = analyze(EXAMPLES / "baseline.txt", list(INSTITUTES))
        self.assertEqual(result.min_support, 2)


class ReportTest(unittest.TestCase):
    def test_markdown_and_json(self) -> None:
        result = analyze(EXAMPLES / "baseline.txt", list(INSTITUTES), min_support=2)
        markdown = to_markdown(
            result.baseline, result.revisions, result.changes,
            result.recommendations, result.min_support,
        )
        self.assertIn("Что предлагается изменить в эталоне", markdown)
        self.assertIn("institute_a", markdown)

        import json

        payload = json.loads(
            to_json(
                result.baseline, result.revisions, result.changes,
                result.recommendations, result.min_support,
            )
        )
        self.assertEqual(len(payload["institutes"]), 3)
        self.assertEqual(len(payload["recommendations"]), len(result.recommendations))
        self.assertTrue(all("coverage" in rec for rec in payload["recommendations"]))


if __name__ == "__main__":
    unittest.main()
