"""Общая обвязка тестов: конфиги во временном каталоге и копии моделей без настоящих процессов."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

import proxy as P

ROOT = Path(tempfile.mkdtemp(prefix="proxy-tests-"))
(ROOT / "model").mkdir(exist_ok=True)
SCRIPT = ROOT / "run.sh"
SCRIPT.write_text("#!/bin/bash\n")

ok = fail = 0


def check(name: str, condition: bool, extra: object = "") -> None:
    """Отмечает проверку пройденной или проваленной."""
    global ok, fail
    if condition:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {extra}")


def report() -> int:
    """Печатает итог и возвращает код выхода."""
    print(f"\nитого: {ok} ок, {fail} провалов")
    return 1 if fail else 0


def one_model(**over) -> dict:
    """Конфиг с одной моделью на одном порту."""
    return {"internal_ports": [9000, 9100], "admin_port": 4999,
            "models": {"m": {"cmd": "vllm serve X", "cwd": str(ROOT / "model"),
                             "vram_gb": 10, "port": 8000}}, **over}


def production(**over) -> dict:
    """Конфиг как на бою: четыре модели по две на порт, одна приоритетная."""
    def model(vram: float, port: int, **extra) -> dict:
        return {"cwd": str(ROOT / "model"), "script": str(SCRIPT),
                "vram_gb": vram, "port": port, **extra}

    return {"internal_ports": [6100, 6199], "admin_port": 4999, "asleep_tail_gb": 0.5,
            "models": {"grade-ai": model(11.2, 6000, priority=True, aliases=["ya-gpt-v2"]),
                       "detector-14b": model(11.2, 6000, aliases=["Qwen/Qwen3-14B-AWQ"]),
                       "test-gen": model(12.8, 5000, aliases=["tasks_v2_r2_merged"]),
                       "detector-4b": model(12.8, 5000, aliases=["Qwen/Qwen3.5-4B"])}, **over}


def load(data: dict) -> P.Settings:
    """Пишет конфиг во временный файл и загружает его."""
    path = ROOT / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return P.Settings.load(path)


def expect_exit(name: str, data: dict, needle: str) -> None:
    """Проверяет, что негодный конфиг отвергнут с понятным сообщением."""
    try:
        load(data)
        check(name, False, "— SystemExit не случился")
    except SystemExit as error:
        check(name, needle in str(error), f"— сообщение: {error}")


class FakeInstance(P.Instance):
    """Копия модели без настоящего процесса vLLM."""

    slow_probe = False     # растягивает опрос карты, чтобы ловить гонки
    tail_real = 1.0        # сколько ГБ «остаётся» на карте после засыпания

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slept: list[bool] = []

    async def start(self) -> None:
        self.process = type("Proc", (), {"returncode": None, "pid": 1})()
        self.state = P.State.AWAKE
        self.awake_since = time.monotonic()

    async def set_sleeping(self, sleeping: bool) -> bool:
        if self.state is not (P.State.AWAKE if sleeping else P.State.ASLEEP):
            return False
        with self.held():
            await asyncio.sleep(0)
            self.state = P.State.ASLEEP if sleeping else P.State.AWAKE
            if sleeping:
                self.tail_gb = FakeInstance.tail_real
            else:
                self.awake_since = time.monotonic()
        self.slept.append(sleeping)
        return True

    async def stop(self) -> None:
        self.process = None
        self.state = P.State.STOPPED
        self.busy = 0
        self.leaving = False

    def devices_seen(self) -> set[str]:
        return set()


def cluster(settings: P.Settings, cards: int = 2, card_gb: float = 16.0) -> P.Cluster:
    """Кластер на фейковых картах: свободная память считается по тому, что держат копии."""
    async def probe():
        if FakeInstance.slow_probe:
            await asyncio.sleep(0.05)
        return {g: P.Memory(card_gb, card_gb - (sum(i.holds_gb for i in built.gpus[g].instances)
                                                if built.gpus.get(g) else 0))
                for g in range(cards)}

    built = P.Cluster(settings, probe=probe, factory=FakeInstance, control=object())
    built._owns_control = False
    return built


def layout(built: P.Cluster) -> str:
    """Раскладка моделей по картам одной строкой."""
    return " | ".join(
        f"GPU{g}({built.gpus[g].memory.free_gb:4.1f}): "
        + (", ".join(f"{i.spec.name}:{i.state.value}" for i in built.gpus[g].instances) or "пусто")
        for g in sorted(built.gpus))
