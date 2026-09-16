"""Необязательный слой: Claude формулирует человеческое объяснение и готовую редакцию пункта.

Работает, если установлен пакет `anthropic` и задан ANTHROPIC_API_KEY.
Без него агент полностью функционален — просто рекомендации остаются в «сыром» виде.
"""

from __future__ import annotations

import json
import os
import textwrap

from .recommend import Recommendation

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = textwrap.dedent(
    """
    Ты — юрист договорного отдела головной организации. Тебе дают паттерн: правку,
    которую несколько подведомственных институтов независимо внесли в типовой договор.
    Твоя задача — помочь обновить типовой договор.

    Верни строго JSON вида:
    {"summary": "...", "wording": "...", "risk": "..."}
    где
      summary — 1-2 предложения: что именно институты меняют и зачем;
      wording — готовая формулировка пункта для типового договора на юридическом русском,
                без номера пункта, максимально нейтральная к обеим сторонам;
      risk     — 1 предложение: на что обратить внимание юристу перед включением.
    Не выдумывай реквизитов, сумм и сроков, которых нет в исходных текстах.
    """
).strip()


def _client(api_key: str | None = None):
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            "Не установлен пакет anthropic. Установите `pip install anthropic` "
            "или запускайте агента без флага --llm."
        ) from exc
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Не задан ANTHROPIC_API_KEY — запустите без --llm или добавьте ключ.")
    return Anthropic(api_key=key)


def _prompt(recommendation: Recommendation) -> str:
    examples = "\n\n".join(
        f"[{institute}] {text}" for institute, text in recommendation.examples
    )
    phrases = "\n".join(f"- {phrase}" for phrase in recommendation.common_phrases)
    return textwrap.dedent(
        f"""
        Тема паттерна: {recommendation.title}
        Тип правки: {recommendation.action}
        Частота: {recommendation.support} из {recommendation.total_institutes} институтов
        Пункт типового договора: {recommendation.target}

        Совпадающие дословно формулировки:
        {phrases or "(дословных совпадений нет)"}

        Редакции институтов:
        {examples}
        """
    ).strip()


def enrich(
    recommendations: list[Recommendation],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_items: int = 20,
) -> list[Recommendation]:
    """Дополнить рекомендации формулировками от Claude (модифицирует объекты на месте)."""
    client = _client(api_key)
    for recommendation in recommendations[:max_items]:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _prompt(recommendation)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            payload = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            recommendation.llm_summary = text
            continue
        recommendation.llm_summary = str(payload.get("summary", "")).strip()
        recommendation.llm_wording = str(payload.get("wording", "")).strip()
        risk = str(payload.get("risk", "")).strip()
        if risk:
            recommendation.llm_summary = f"{recommendation.llm_summary}\n\nРиск: {risk}".strip()
    return recommendations
