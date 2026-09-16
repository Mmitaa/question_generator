"""Извлечение текста из документов договоров.

Поддерживаются .docx (без внешних зависимостей, через zip+XML), .txt/.md,
а также .pdf и .doc — при наличии соответствующих утилит/библиотек.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ExtractionError(RuntimeError):
    """Не удалось прочитать документ."""


def _docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in ("word/document.xml",) if n in zf.namelist()]
        if not names:
            raise ExtractionError(f"{path}: это не похоже на .docx (нет word/document.xml)")
        xml = zf.read(names[0])
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for block in root.iter():
        if block.tag == f"{W_NS}p":
            parts: list[str] = []
            for node in block.iter():
                if node.tag == f"{W_NS}t" and node.text:
                    parts.append(node.text)
                elif node.tag == f"{W_NS}tab":
                    parts.append(" ")
                elif node.tag in (f"{W_NS}br", f"{W_NS}cr"):
                    parts.append(" ")
            text = "".join(parts).strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def _pdf_text(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pass
    else:
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        pass
    else:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if shutil.which("pdftotext"):
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout

    raise ExtractionError(
        f"{path}: для чтения PDF нужен pdfplumber, pypdf или утилита pdftotext. "
        "Либо сохраните договор в .docx/.txt."
    )


def _doc_text(path: Path) -> str:
    for tool, args in (("antiword", []), ("catdoc", [])):
        if shutil.which(tool):
            out = subprocess.run([tool, *args, str(path)], capture_output=True, text=True, check=False)
            if out.returncode == 0:
                return out.stdout
    if shutil.which("libreoffice"):
        raise ExtractionError(
            f"{path}: формат .doc. Конвертируйте в .docx: "
            f"libreoffice --headless --convert-to docx '{path}'"
        )
    raise ExtractionError(f"{path}: формат .doc не поддерживается, сохраните договор в .docx")


def extract_paragraphs(path: str | Path) -> list[str]:
    """Вернуть список абзацев документа."""
    path = Path(path)
    if not path.exists():
        raise ExtractionError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()
    if suffix == ".docx":
        paragraphs = _docx_paragraphs(path)
    elif suffix in (".txt", ".md"):
        paragraphs = path.read_text(encoding="utf-8", errors="replace").splitlines()
    elif suffix == ".pdf":
        paragraphs = _pdf_text(path).splitlines()
    elif suffix == ".doc":
        paragraphs = _doc_text(path).splitlines()
    else:
        raise ExtractionError(f"{path}: неизвестный формат «{suffix}»")

    cleaned = [re.sub(r"[ \t\xa0]+", " ", p).strip() for p in paragraphs]
    return [p for p in cleaned if p]
