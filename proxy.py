"""Прокси перед несколькими vLLM: усыпляет, будит и переносит модели между картами, а на время обучения отдаёт карты ему. Запуск: python proxy.py"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import itertools
import json
import logging
import logging.handlers
import os
import re
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, NoReturn

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

HERE = Path(__file__).parent
log = logging.getLogger("proxy")

SAFE_PATH = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")   # без точек и процентов: через них видны служебные ручки vLLM
PORT_STEP = 10                  # vLLM занимает порты сразу за своим, поэтому раздаём с запасом
RESIDENCY_GRACE = 3.0           # столько секунд без запросов — и свежую копию уже не защищаем
STREAM_BUFFER = 32 * 2 ** 20    # столько байт стрима копим медленному клиенту, дальше рвём его
GIB = 1024 ** 3
PROC_SCAN_LIMIT = 2.0           # потолок паузы между обходами /proc

STRIPPED_ENV = ("VIRTUAL_ENV", "POETRY_ACTIVE", "PYTHONHOME", "PYTHONPATH")
MODEL_KEYS = frozenset({"cwd", "script", "vram_gb", "port", "aliases", "priority", "env"})
TRAINING_KEYS = frozenset({"gpus", "pid", "owner", "vram_gb"})


class NoRoom(RuntimeError):
    """Модель не влезает ни на одну карту даже после вытеснения соседей."""


class TrainingInProgress(NoRoom):
    """Все карты отданы под обучение."""


def fail(message: str) -> NoReturn:
    """Останавливает запуск понятным сообщением вместо traceback."""
    raise SystemExit(message)


# ── конфигурация ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """Описание одной модели из конфига."""

    name: str
    cwd: Path
    script: Path
    vram_gb: float
    port: int
    aliases: tuple[str, ...] = ()
    priority: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, raw: dict) -> ModelSpec:
        """Собирает описание модели из секции конфига и проверяет его."""
        if unknown := sorted(set(raw or ()) - MODEL_KEYS):
            fail(f"{name}: непонятные поля {', '.join(unknown)}. "
                 f"Допустимы: {', '.join(sorted(MODEL_KEYS))}")
        try:
            spec = cls(name=name, cwd=Path(raw["cwd"]), script=Path(raw["script"]),
                       vram_gb=float(raw["vram_gb"]), port=int(raw["port"]),
                       aliases=tuple(raw.get("aliases", ())),
                       priority=bool(raw.get("priority")),
                       env={k: str(v) for k, v in (raw.get("env") or {}).items()})
        except KeyError as error:
            fail(f"{name}: в конфиге нет обязательного поля {error.args[0]}")
        except (TypeError, ValueError, AttributeError) as error:
            fail(f"{name}: неверное значение в конфиге — {error}")
        spec.validate()
        return spec

    def validate(self) -> None:
        """Проверяет пути и объём; сообщает обо всех ошибках разом."""
        broken = [message for bad, message in (
            (not self.script.is_file(), f"скрипта {self.script} нет"),
            (not self.cwd.is_dir(), f"каталога {self.cwd} нет"),
            (self.vram_gb <= 0, "vram_gb должно быть больше нуля"),
        ) if bad]
        if broken:
            fail(f"{self.name}: " + "; ".join(broken))


@dataclass(frozen=True)
class Settings:
    """Весь конфиг прокси; создаётся через Settings.load()."""

    models: dict[str, ModelSpec]
    internal_ports: range
    env: dict[str, str] = field(default_factory=dict)
    admin_port: int = 4999
    api_key: str | None = None
    reserve_gb: float = 1.0
    asleep_tail_gb: float = 0.5
    idle_sleep_sec: float = 120
    idle_stop_sec: float = 3600
    min_residency_sec: float = 30
    start_timeout: float = 900
    switch_timeout: float = 120
    queue_timeout: float = 180
    drain_timeout: float = 600
    read_timeout: float | None = None
    log_keep: int = 20
    preload: bool = True
    vllm_log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Строит индексы для поиска модели и проверяет конфиг на противоречия."""
        object.__setattr__(self, "vllm_log_level", self.vllm_log_level.upper())
        object.__setattr__(self, "_by_port", self._index_ports())
        object.__setattr__(self, "_by_alias", self._index_aliases())
        self._check_limits()

    def _index_ports(self) -> dict[int, list[str]]:
        """Группирует модели по внешним портам."""
        by_port: dict[int, list[str]] = {}
        for name, spec in self.models.items():
            if spec.port in self.internal_ports or spec.port == self.admin_port:
                fail(f"{name}: внешний порт {spec.port} пересекается со служебными")
            by_port.setdefault(spec.port, []).append(name)
        return by_port

    def _index_aliases(self) -> dict[str, str]:
        """Строит карту «имя или алиас -> модель»."""
        by_alias: dict[str, str] = {}
        for name, spec in self.models.items():
            for alias in (name, *spec.aliases):
                if alias in by_alias:
                    fail(f"имя {alias!r} занято двумя моделями: {by_alias[alias]} и {name}")
                by_alias[alias] = name
        return by_alias

    def _check_limits(self) -> None:
        """Проверяет числовые настройки; сообщает обо всех ошибках разом."""
        broken = [message for bad, message in (
            (not self.models, "в конфиге нет ни одной модели"),
            (not self.internal_ports, "internal_ports пуст: укажите [низ, верх]"),
            (min(self.reserve_gb, self.asleep_tail_gb) < 0,
             "reserve_gb и asleep_tail_gb не могут быть отрицательными"),
            (self.min_residency_sec >= self.queue_timeout,
             "min_residency_sec должен быть меньше queue_timeout, иначе ждущие не дождутся карты"),
            (self.admin_port in self.internal_ports,
             f"admin_port {self.admin_port} попал в internal_ports"),
            (len(self.internal_ports) < len(self.models),
             f"internal_ports вмещает {len(self.internal_ports)} портов, "
             f"а моделей {len(self.models)}"),
        ) if bad]
        if broken:
            fail("; ".join(broken))

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        """Читает config.yaml и собирает из него Settings."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            fail(f"{path}: ожидается словарь с настройками")
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - known):
            fail(f"{path}: непонятные поля {', '.join(unknown)}. "
                 f"Допустимы: {', '.join(sorted(known))}")
        for required in ("models", "internal_ports"):
            if required not in data:
                fail(f"{path}: в конфиге нет обязательного поля {required}")
        try:
            low, high = (int(edge) for edge in data["internal_ports"])
        except (TypeError, ValueError):
            fail(f"{path}: internal_ports должен быть парой чисел [низ, верх]")
        if not isinstance(data["models"], dict):
            fail(f"{path}: models должен быть словарём «имя: настройки»")
        skip = {"models", "internal_ports", "env"}
        return cls(models={name: ModelSpec.from_config(name, raw)
                           for name, raw in data["models"].items()},
                   internal_ports=range(low, high + 1),
                   env={k: str(v) for k, v in (data.get("env") or {}).items()},
                   **{f.name: data[f.name] for f in fields(cls)
                      if f.name in data and f.name not in skip})

    @property
    def listen(self) -> list[int]:
        """Внешние порты, на которых поднимаем HTTP."""
        return sorted(self._by_port)

    @property
    def wait_budget(self) -> float:
        """Потолок ожидания запроса: очередь за местом плюс один чужой холодный старт."""
        return self.queue_timeout + self.start_timeout

    def on_port(self, port: int) -> list[str]:
        """Модели, закреплённые за портом."""
        return self._by_port.get(port, [])

    def resolve(self, requested: str | None, port: int) -> str | None:
        """Определяет модель по полю model; на одномодельном порту поле не важно."""
        here = self.on_port(port)
        if len(here) == 1:
            return here[0]
        name = self._by_alias.get(requested) if isinstance(requested, str) else None
        return name if name in here else None


# ── мелкие утилиты ─────────────────────────────────────────────────────────

def strip_venv_from_path(path: str, virtual_env: str | None) -> str:
    """Убирает из PATH venv самой прокси, иначе poetry в инстансе возьмёт его вместо проектного."""
    if not virtual_env:
        return path
    unwanted = str(Path(virtual_env) / "bin")
    return os.pathsep.join(part for part in path.split(os.pathsep) if part != unwanted)


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """Свободен ли порт на том адресе, на котором мы собираемся слушать."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def new_log(folder: Path, prefix: str, keep: int = 20) -> Path:
    """Заводит файл лога с датой в имени, переводит на него симлинк latest и чистит старые."""
    folder.mkdir(parents=True, exist_ok=True)
    old = sorted(folder.glob(f"{prefix}_20*"))
    for stale in old[:max(0, len(old) - keep + 1)]:
        with contextlib.suppress(OSError):
            stale.unlink()
    path = folder / f"{prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    with contextlib.suppress(OSError):
        link = folder / f"{prefix}_latest.log"
        link.unlink(missing_ok=True)
        link.symlink_to(path.name)
    return path


async def wait_unlocked(lock: asyncio.Lock, deadline: float, what: str) -> None:
    """Ждёт освобождения лока карты, сам его не беря."""
    while lock.locked():
        if time.monotonic() > deadline:
            raise NoRoom(f"не дождались, пока {what}")
        await asyncio.sleep(0.2)


# ── взгляд на железо ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Memory:
    """Сколько памяти всего и сколько свободно на карте."""
    total_gb: float
    free_gb: float


Probe = Callable[[], Awaitable[dict[int, Memory]]]
_nvml = {"usable": True, "ready": False}


async def read_gpu_memory() -> dict[int, Memory]:
    """Читает свободную память карт: сначала через pynvml, при неудаче — через nvidia-smi."""
    if _nvml["usable"]:
        try:
            return await asyncio.to_thread(_nvml_memory)
        except Exception as error:
            _nvml["usable"] = False
            log.warning("pynvml недоступен (%s), читаю память через nvidia-smi. "
                        "Поставьте nvidia-ml-py в окружение прокси", error)
    return await _smi_memory()


def _nvml_memory() -> dict[int, Memory]:
    """Читает память карт через pynvml."""
    import pynvml

    if not _nvml["ready"]:
        pynvml.nvmlInit()
        _nvml["ready"] = True
    return {index: Memory(info.total / GIB, info.free / GIB)
            for index in range(pynvml.nvmlDeviceGetCount())
            for info in [pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(index))]}


async def _smi_memory() -> dict[int, Memory]:
    """Читает память карт через nvidia-smi; без него размещать вслепую нельзя."""
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await process.communicate()
        if process.returncode != 0:
            raise OSError(f"nvidia-smi вернул {process.returncode}")
        rows = [list(map(int, line.split(","))) for line in out.decode().strip().splitlines()]
        return {index: Memory(total / 1024, free / 1024) for index, total, free in rows}
    except (OSError, ValueError) as error:
        raise NoRoom(f"ни pynvml, ни nvidia-smi недоступны ({error}) — размещать вслепую нельзя")


# ── процессы ───────────────────────────────────────────────────────────────

def proc_stat(pid: int) -> tuple[str, str] | None:
    """Состояние процесса и метку его старта из /proc; None, если процесса нет."""
    with contextlib.suppress(OSError, IndexError, ValueError):
        columns = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return columns[0], columns[19]
    return None


def group_members(pgid: int) -> list[int]:
    """Живые процессы группы; зомби не считаем — память карты они уже отдали."""
    members = []
    for proc in Path("/proc").glob("[0-9]*"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            state, _, group = (proc / "stat").read_text().rsplit(")", 1)[1].split()[:3]
            if int(group) == pgid and state != "Z":
                members.append(int(proc.name))
    return members


def orphan_groups(settings: Settings) -> dict[int, str]:
    """Находит группы vLLM от прошлого запуска прокси по метке в окружении."""
    groups: dict[int, str] = {}
    mine = os.getpgid(0)
    for proc in Path("/proc").glob("[0-9]*"):
        with contextlib.suppress(OSError, ValueError):
            env = dict(item.split(b"=", 1)
                       for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            name, _, port = env.get(b"PROXY_INSTANCE", b"").decode().rpartition(":")
            if not name:          # запуск от старой версии прокси
                name = env.get(b"VLLM_SERVED_NAME", b"").decode()
                port = env.get(b"VLLM_PORT", b"0").decode()
            pgid = os.getpgid(int(proc.name))
            if name in settings.models and int(port) in settings.internal_ports and pgid != mine:
                groups[pgid] = name
    return groups


async def kill_groups(pgids: Iterable[int], grace: float = 15) -> None:
    """Просит группы завершиться, через grace секунд добивает оставшиеся."""
    alive = set(pgids)
    for pgid in alive:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while alive and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        for pgid in list(alive):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                alive.discard(pgid)
    for pgid in alive:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


async def kill_orphans(settings: Settings) -> list[str]:
    """Гасит хвосты прошлого запуска, чтобы они не держали память карт."""
    groups = await asyncio.to_thread(orphan_groups, settings)
    await kill_groups(groups)
    return [f"{name} (группа {pgid})" for pgid, name in groups.items()]


# ── копия модели ───────────────────────────────────────────────────────────

class State(Enum):
    """Состояние процесса vLLM; спящий держит веса в RAM, а на карте только CUDA-контекст."""
    STOPPED = "остановлена"
    STARTING = "запускается"
    AWAKE = "активна"
    ASLEEP = "спит"


class Instance:
    """Одна копия модели — один процесс vLLM."""

    def __init__(self, spec: ModelSpec, gpu_id: int, port: int, settings: Settings,
                 client: httpx.AsyncClient):
        self.spec = spec
        self.gpu_id = gpu_id
        self.port = port
        self.settings = settings
        self.client = client
        self.state = State.STOPPED
        self.busy = 0
        self.leaving = False
        self.can_sleep = True                # сбрасываем, если vLLM не смог уснуть
        self.awake_gb: float | None = None   # замеры; пока их нет, считаем по конфигу
        self.tail_gb: float | None = None
        self.last_used = time.monotonic()
        self.awake_since = 0.0
        self.process: asyncio.subprocess.Process | None = None

    def __repr__(self) -> str:
        return f"<{self.spec.name} GPU{self.gpu_id}:{self.port} {self.state.value}>"

    @property
    def url(self) -> str:
        """Локальный адрес, на котором слушает vLLM."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        """Процесс запущен и ещё не завершился."""
        return self.process is not None and self.process.returncode is None

    @property
    def idle_for(self) -> float:
        """Сколько секунд копия простаивает; под нагрузкой ноль."""
        return 0.0 if self.busy else time.monotonic() - self.last_used

    @property
    def footprint(self) -> float:
        """Сколько памяти копия занимает активной."""
        return self.awake_gb if self.awake_gb is not None else self.spec.vram_gb

    @property
    def tail(self) -> float:
        """Сколько памяти копия держит во сне."""
        return self.tail_gb if self.tail_gb is not None else self.settings.asleep_tail_gb

    @property
    def holds_gb(self) -> float:
        """Сколько памяти копия занимает на карте прямо сейчас."""
        return {State.AWAKE: self.footprint, State.ASLEEP: self.tail}.get(self.state, 0.0)

    @property
    def frees_gb(self) -> float:
        """Сколько памяти освободится при выселении; активную обычно усыпляют."""
        if self.state is State.AWAKE and self.can_sleep:
            return max(0.0, self.footprint - self.tail)
        return self.holds_gb

    @property
    def wake_need(self) -> float:
        """Сколько памяти займёт пробуждение — ровно то, что копия отдала при засыпании."""
        return max(0.0, self.footprint - self.tail)

    @contextlib.contextmanager
    def held(self):
        """Закрывает вход на время перехода и возвращает флаг как было: его мог выставить кто-то ещё."""
        was_leaving = self.leaving
        self.leaving = True
        try:
            yield
        finally:
            self.leaving = was_leaving

    def devices_seen(self) -> set[str]:
        """Какие CUDA_VISIBLE_DEVICES на деле видят процессы копии."""
        seen = set()
        for pid in group_members(self.process.pid) if self.process else []:
            with contextlib.suppress(OSError):
                for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                        seen.add(item.split(b"=", 1)[1].decode())
        return seen

    def recency(self, now: float) -> float:
        """Когда копия работала в последний раз; занятая работает сейчас."""
        return now if self.busy else self.last_used

    def resident(self, now: float) -> bool:
        """Проснулась недавно и ею пользуются — такую не выселяем, иначе модели начнут перекидывать карту."""
        return (self.state is State.AWAKE
                and now - self.awake_since < self.settings.min_residency_sec
                and (self.busy > 0 or now - self.last_used < RESIDENCY_GRACE))

    def reserve(self) -> bool:
        """Занимает копию под запрос. Без await: между проверкой и захватом никто не влезет."""
        if self.state is State.AWAKE and self.alive and not self.leaving:
            self.busy += 1
            return True
        return False

    def release(self) -> None:
        """Отпускает копию и запоминает время для LRU."""
        self.busy = max(0, self.busy - 1)
        self.last_used = time.monotonic()

    # ── жизненный цикл ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Запускает процесс, ждёт /health и прогревает модель."""
        path = new_log(self.spec.cwd / "logs", self.spec.name, self.settings.log_keep)
        self.state = State.STARTING
        log.info("запускаю %s на GPU%s, порт %s, лог %s", self.spec.name, self.gpu_id, self.port, path)

        with open(path, "wb") as sink:
            # отдельная группа, чтобы при остановке убить и bash-обёртку, и сам vLLM
            self.process = await asyncio.create_subprocess_exec(
                "bash", str(self.spec.script), cwd=self.spec.cwd, env=self.environment(), start_new_session=True,
                stdout=sink, stderr=asyncio.subprocess.STDOUT)

        deadline = time.monotonic() + self.settings.start_timeout
        while time.monotonic() < deadline:
            if not self.alive:
                raise RuntimeError(f"{self.spec.name}: процесс умер при старте, см. {path}")
            if await self.responds():
                self.state = State.AWAKE
                self.awake_since = time.monotonic()
                await self.warm_up()
                log.info("%s поднялась", self.spec.name)
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"{self.spec.name}: не поднялась за "
                           f"{self.settings.start_timeout:.0f} сек, см. {path}")

    def environment(self) -> dict[str, str]:
        """Собирает окружение процесса: без следов venv прокси, с настройками модели и vLLM."""
        env = {key: value for key, value in os.environ.items() if key not in STRIPPED_ENV}
        env["PATH"] = strip_venv_from_path(env.get("PATH", ""), os.environ.get("VIRTUAL_ENV"))
        env.update(self.settings.env)
        env.update(self.spec.env)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # нумерация как в NVML, иначе нулевой окажется другая карта
        env["VLLM_SERVER_DEV_MODE"] = "1"         # без этого vLLM не отдаёт /sleep и /wake_up
        env["PROXY_INSTANCE"] = f"{self.spec.name}:{self.port}"   # по метке следующий запуск найдёт хвосты
        env["VLLM_SERVED_NAME"] = self.spec.name
        env["VLLM_LOGGING_LEVEL"] = self.settings.vllm_log_level
        env["VLLM_PORT"] = str(self.port)         # скрипт обязан отдать его vLLM флагом --port
        return env

    async def stop(self) -> None:
        """Останавливает процесс: сначала мягко, через 30 секунд убивает."""
        self.leaving = True
        if self.alive:
            log.info("останавливаю %s", self.spec.name)
            self.signal_group(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except asyncio.TimeoutError:
                log.warning("%s не завершилась за 30 сек — убиваю", self.spec.name)
                self.signal_group(signal.SIGKILL)
                await self.process.wait()
        if self.process is not None:
            await self.wait_group(self.process.pid)
        self.process = None
        self.state = State.STOPPED
        self.busy = 0
        self.leaving = False

    async def wait_group(self, pgid: int) -> None:
        """Ждёт выхода всех процессов копии: пока жив хоть один, память карты занята."""
        deadline = time.monotonic() + 15
        delay = 0.2
        while await asyncio.to_thread(group_members, pgid):   # обход /proc не держим в event loop
            if time.monotonic() > deadline:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)
                await asyncio.sleep(0.5)
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, PROC_SCAN_LIMIT)

    def signal_group(self, sig: int) -> None:
        """Шлёт сигнал всей группе, а если её уже нет — самому процессу."""
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                self.process.send_signal(sig)

    # ── сон и пробуждение ──────────────────────────────────────────────────

    async def set_sleeping(self, sleeping: bool) -> bool:
        """Усыпляет или будит модель; False — если переходить было не из чего."""
        if self.state is not (State.AWAKE if sleeping else State.ASLEEP):
            return False
        started = time.monotonic()
        log.info("%s %s", "усыпляю" if sleeping else "бужу", self.spec.name)
        with self.held():
            await self._switch(sleeping)
            await self.await_state(sleeping)
            self.state = State.ASLEEP if sleeping else State.AWAKE
            log.info("%s %s за %.1f сек", self.spec.name,
                     "уснула" if sleeping else "проснулась", time.monotonic() - started)
            if not sleeping:
                self.awake_since = time.monotonic()
                await self.warm_up()
        return True

    async def _switch(self, sleeping: bool) -> None:
        """Дёргает ручку сна или пробуждения у vLLM."""
        endpoint = "/sleep?level=1" if sleeping else "/wake_up"
        response = await self.client.post(f"{self.url}{endpoint}",
                                          timeout=self.settings.switch_timeout)
        if response.status_code == 404:
            raise RuntimeError(f"у vLLM нет {endpoint.split('?')[0]}: запущен без "
                               f"VLLM_SERVER_DEV_MODE=1, слишком старый или за портом не vLLM")
        if response.status_code != 200:
            raise RuntimeError(f"{endpoint} вернул {response.status_code}: {response.text[:300]}")

    async def await_state(self, sleeping: bool) -> None:
        """Опрашивает /is_sleeping, пока состояние не сменится; не-JSON в ответе просто ждём дальше."""
        deadline = time.monotonic() + self.settings.switch_timeout
        while True:
            with contextlib.suppress(httpx.HTTPError, ValueError, AttributeError):
                response = await self.client.get(f"{self.url}/is_sleeping")
                if bool(response.json().get("is_sleeping", False)) is sleeping:
                    return
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.spec.name}: состояние не сменилось за "
                                   f"{self.settings.switch_timeout:.0f} сек")
            await asyncio.sleep(0.25)

    async def responds(self) -> bool:
        """Отвечает ли vLLM на /health."""
        try:
            return (await self.client.get(f"{self.url}/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def warm_up(self) -> None:
        """Шлёт короткий запрос, чтобы vLLM собрал CUDA-графы заранее."""
        try:
            response = await self.client.post(
                f"{self.url}/v1/completions", timeout=180,
                json={"model": self.spec.name, "prompt": "ok", "max_tokens": 1})
            if response.status_code != 200:
                log.warning("прогрев %s не удался: vLLM ответил %s (%s). Для моделей только под чат "
                            "или эмбеддинги это нормально, но первый запрос будет медленным",
                            self.spec.name, response.status_code, response.text[:200])
        except httpx.HTTPError as error:
            log.warning("прогрев %s не удался: %s", self.spec.name, error)


# ── карта ──────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Gpu:
    """Одна видеокарта с её памятью и живущими на ней копиями."""

    id: int
    memory: Memory
    reserve_gb: float
    instances: list[Instance] = field(default_factory=list)
    blocked: set[str] = field(default_factory=set)          # аренды обучения
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)   # одно переключение за раз

    def __repr__(self) -> str:
        note = ", отдана под обучение" if self.blocked else ""
        return (f"GPU{self.id} свободно {self.memory.free_gb:.1f} "
                f"из {self.memory.total_gb:.0f} ГБ{note}")

    def foreign_gb(self) -> float:
        """Сколько памяти держат не наши процессы."""
        return max(0.0, self.memory.total_gb - self.memory.free_gb
                   - sum(i.holds_gb for i in self.instances))

    def required(self, need: float) -> float:
        """Сколько свободной памяти нужно под need ГБ вместе с резервом."""
        return need + self.reserve_gb

    def fits(self, need: float) -> bool:
        """Влезет ли прямо сейчас ещё need ГБ."""
        return self.memory.free_gb >= self.required(need)

    def crowded(self, newcomer: tuple[float, float], leaving: list[Instance]) -> bool:
        """Сможет ли каждый здешний житель проснуться, пока остальные спят рядом."""
        members = [(i.footprint, i.tail) for i in self.instances
                   if i.state is not State.STOPPED and i not in leaving] + [newcomer]
        tails = sum(tail for _, tail in members)
        room = self.memory.total_gb - self.foreign_gb() - self.reserve_gb
        return any(footprint + tails - tail > room for footprint, tail in members)

    def evictable(self, spec: ModelSpec, need: float, wait: bool = False) -> list[Instance] | None:
        """Кого выселить под need ГБ: сначала давно не нужных, приоритетных последними, никого лишнего."""
        short = self.required(need) - self.memory.free_gb
        if short <= 0:
            return []
        now = time.monotonic()
        candidates = sorted((i for i in self.instances
                             if i.spec.name != spec.name and i.frees_gb > 0
                             and (wait or (i.busy == 0 and not i.resident(now)))),
                            key=lambda i: (i.spec.priority, i.recency(now)))
        chosen: list[Instance] = []
        freed = 0.0
        for instance in candidates:
            chosen.append(instance)
            freed += instance.frees_gb
            if freed >= short:
                break
        else:
            return None
        for instance in list(chosen):   # спящего соседа не гасим, если место набирается и без него
            if freed - instance.frees_gb >= short:
                chosen.remove(instance)
                freed -= instance.frees_gb
        return chosen


# ── обучение ───────────────────────────────────────────────────────────────

@dataclass
class Lease:
    """Аренда карт под обучение: пока она есть, прокси не ставит на них модели."""

    id: str
    gpus: set[int]
    owner: str
    pid: int | None
    started: str | None = None       # метка старта процесса против переиспользованных pid
    since: float = field(default_factory=time.time)

    def alive(self) -> bool:
        """Жив ли процесс обучения; без pid аренда живёт до явного снятия."""
        if self.pid is None:
            return True
        stat = proc_stat(self.pid)
        if stat is None or stat[0] == "Z":
            return False
        return self.started is None or stat[1] == self.started

    def describe(self) -> dict:
        """Аренда в виде словаря для HTTP-ответов."""
        return {"id": self.id, "owner": self.owner, "gpus": sorted(self.gpus), "pid": self.pid,
                "since": datetime.fromtimestamp(self.since).isoformat(timespec="seconds")}


# ── кластер ────────────────────────────────────────────────────────────────

class Cluster:
    """Все карты и все копии: решает, что куда поставить, кого подвинуть и когда уступить обучению."""

    def __init__(self, settings: Settings, probe: Probe = read_gpu_memory,
                 factory: type[Instance] = Instance,
                 control: httpx.AsyncClient | None = None):
        self.settings = settings
        self.probe = probe
        self.factory = factory
        self.control = control or httpx.AsyncClient(timeout=30)
        self._owns_control = control is None
        self.gpus: dict[int, Gpu] = {}
        self.instances: dict[str, Instance] = {}
        self.model_locks = {name: asyncio.Lock() for name in settings.models}
        self.probe_lock = asyncio.Lock()
        self.probed_at = 0.0
        self.leases: dict[str, Lease] = {}
        self.footprints: dict[str, float] = {}   # замеры по моделям, чтобы новая копия считалась верно
        self.tails: dict[str, float] = {}
        self.crowd_warned: set[str] = set()
        self.tasks: set[asyncio.Task] = set()

    def spawn(self, coro: Awaitable[Any]) -> None:
        """Запускает фоновую задачу, держа на неё ссылку, чтобы её не собрал GC."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # ── доступ к моделям ───────────────────────────────────────────────────

    async def acquire(self, name: str) -> Instance:
        """Возвращает готовую копию, уже занятую под запрос; отпускать через release()."""
        if instance := self.ready(name):
            return instance
        deadline = time.monotonic() + self.settings.wait_budget
        lock = self.model_locks[name]
        try:
            await asyncio.wait_for(lock.acquire(), self.settings.wait_budget)
        except asyncio.TimeoutError:
            raise NoRoom(f"{name}: не дождались очереди за {self.settings.wait_budget:.0f} сек, "
                         f"всё это время модель поднимал другой запрос")
        try:
            return self.ready(name) or await self.ensure_ready(name, deadline)
        finally:
            lock.release()

    def ready(self, name: str) -> Instance | None:
        """Занимает копию модели, если она уже поднята и свободна."""
        instance = self.instances.get(name)
        return instance if instance is not None and instance.reserve() else None

    async def ensure_ready(self, name: str, deadline: float) -> Instance:
        """Будит спящую копию или запускает новую и занимает её. Вызывать под model_locks[name]."""
        spec = self.settings.models[name]
        while time.monotonic() <= deadline:
            await self.refresh()
            if current := self.instances.get(name):
                await self.reap_dead(current)
            if instance := self.ready(name):
                return instance
            if await self.wait_moving(spec, deadline) or await self.drop_stale(spec):
                continue
            gpu, sleeper = self.plan(spec)
            if gpu.lock.locked():
                await wait_unlocked(gpu.lock, deadline, f"закончится переключение на GPU{gpu.id}")
                continue
            async with gpu.lock:
                if instance := await self.place(spec, gpu, sleeper, deadline):
                    return instance
        raise NoRoom(f"{name}: не дождались места за {self.settings.wait_budget:.0f} сек")

    async def place(self, spec: ModelSpec, gpu: Gpu, sleeper: Instance | None,
                    deadline: float) -> Instance | None:
        """Ставит модель на выбранную карту и занимает её; None — расклад поменялся. Под gpu.lock."""
        try:
            if self.plan(spec, mine=gpu) != (gpu, sleeper):
                return None
        except NoRoom:
            await asyncio.sleep(0.2)     # карту забрали, пока мы брали лок
            return None
        log.info("%s -> GPU%s: %s", spec.name, gpu.id,
                 "бужу спящую копию" if sleeper else "запускаю копию")
        await self.free_up(gpu, spec, self.need_for(spec, sleeper), deadline)
        if sleeper is None:
            instance = await self.launch(gpu, spec)
        elif await self.wake(sleeper):
            instance = sleeper
        else:
            return None                  # не проснулась и уже остановлена, поднимем заново
        return instance if instance.reserve() else None

    async def wait_moving(self, spec: ModelSpec, deadline: float) -> bool:
        """Ждёт, пока копия домучает переключение: дождаться дешевле, чем запускать новую."""
        instance = self.instances.get(spec.name)
        if instance is None or instance.state is State.ASLEEP:
            return False        # дальше plan() рассчитывает, что копии либо нет, либо она спит
        gpu = self.gpus[instance.gpu_id]
        if gpu.lock.locked():
            await wait_unlocked(gpu.lock, deadline, f"{spec.name} закончит переключение")
        else:
            await asyncio.sleep(0.2)
        return True

    def is_stale(self, instance: Instance) -> bool:
        """Спящая копия на карте, где ей уже не проснуться: память забрал чужой процесс."""
        return (instance.state is State.ASLEEP and self.gpus[instance.gpu_id].evictable(
            instance.spec, instance.wake_need, wait=True) is None)

    async def drop_stale(self, spec: ModelSpec) -> bool:
        """Забывает спящую копию, которой уже не проснуться на своей карте."""
        instance = self.instances.get(spec.name)
        if instance is None or not self.is_stale(instance):
            return False
        gpu = self.gpus[instance.gpu_id]
        if gpu.lock.locked():
            return False
        log.warning("%s: на GPU%s свободно %.1f ГБ, для пробуждения нужно %.1f, и подвинуть некого "
                    "— память держит посторонний процесс (nvidia-smi). Запускаю заново там, где "
                    "есть место", spec.name, gpu.id, gpu.memory.free_gb,
                    gpu.required(instance.wake_need))
        async with gpu.lock:
            await self.forget(instance)
        return True

    # ── выбор карты ────────────────────────────────────────────────────────

    def need_for(self, spec: ModelSpec, sleeper: Instance | None) -> float:
        """Сколько памяти займёт модель: спящей — что она отдала, новой — не меньше vram_gb."""
        if sleeper is not None:
            return sleeper.wake_need
        return max(spec.vram_gb, self.footprints.get(spec.name, 0.0))

    def crowds(self, gpu: Gpu, spec: ModelSpec, victims: list[Instance]) -> bool:
        """Станет ли тесно на карте от новой копии; тех, кого остановят, не считаем."""
        gone = [v for v in victims if v.state is State.ASLEEP or not v.can_sleep]
        tail = self.tails.get(spec.name, self.settings.asleep_tail_gb)
        return gpu.crowded((self.need_for(spec, None), tail), gone)

    def plan(self, spec: ModelSpec, mine: Gpu | None = None) -> tuple[Gpu, Instance | None]:
        """Выбирает карту для модели; разбудить спящую выгоднее, чем запускать новую."""
        options = self.candidates(spec, mine)
        if not options:
            raise self.nowhere(spec)
        key, _, gpu, sleeper = min(options, key=lambda option: option[:2])
        if key[0] and spec.name not in self.crowd_warned:      # key[0] — признак тесноты
            self.crowd_warned.add(spec.name)
            log.warning("%s: ни на одной карте нет места держать её спящей рядом с остальными. "
                        "При переключениях соседа придётся останавливать, и его следующий запрос "
                        "будет ждать холодного старта", spec.name)
        return gpu, sleeper

    def candidates(self, spec: ModelSpec, mine: Gpu | None) -> list[tuple]:
        """Оценивает каждую карту под модель; чем меньше ключ, тем лучше вариант."""
        # сюда попадаем, только когда копии нет или она спит: остальное отсеял wait_moving
        sleeper = self.instances.get(spec.name)
        now = time.monotonic()
        options = []
        for gpu in ([self.gpus[sleeper.gpu_id]] if sleeper else list(self.gpus.values())):
            if gpu.blocked:
                continue
            need = self.need_for(spec, sleeper)
            victims = gpu.evictable(spec, need)
            must_wait = victims is None
            if victims is None and (victims := gpu.evictable(spec, need, wait=True)) is None:
                continue
            key = (sleeper is None and self.crowds(gpu, spec, victims),
                   gpu.lock.locked() and gpu is not mine,
                   sum(v.spec.priority for v in victims),
                   must_wait, len(victims),
                   max((v.recency(now) for v in victims), default=0.0),
                   -gpu.memory.free_gb)
            options.append((key, gpu.id, gpu, sleeper))
        return options

    def nowhere(self, spec: ModelSpec) -> NoRoom:
        """Объясняет, почему модель некуда поставить."""
        if self.gpus and all(gpu.blocked for gpu in self.gpus.values()):
            return TrainingInProgress(f"{spec.name}: все карты отданы под обучение, "
                                      f"модели вернутся, когда оно закончится")
        return NoRoom(f"{spec.name}: {spec.vram_gb:.1f} ГБ не найдётся нигде. "
                      + "; ".join(map(repr, self.gpus.values())))

    # ── память карт ────────────────────────────────────────────────────────

    async def refresh(self, max_age: float = 1.0) -> None:
        """Обновляет данные о памяти карт, если последний опрос был давно."""
        if time.monotonic() - self.probed_at < max_age:
            return
        async with self.probe_lock:
            if time.monotonic() - self.probed_at < max_age:
                return
            for index, memory in (await self.probe()).items():
                if gpu := self.gpus.get(index):
                    gpu.memory = memory
                else:
                    self.gpus[index] = Gpu(index, memory, self.settings.reserve_gb)
            self.probed_at = time.monotonic()

    def invalidate(self) -> None:
        """Сбрасывает кэш памяти после того, как мы сами что-то запустили или остановили."""
        self.probed_at = 0.0

    async def free_now(self, gpu: Gpu) -> float:
        """Свежий замер свободной памяти на карте."""
        await self.refresh(max_age=0)
        return gpu.memory.free_gb

    def note_awake(self, instance: Instance, gpu: Gpu, base: float, grown: float) -> None:
        """Запоминает, сколько копия занимает активной; замер, разошедшийся с конфигом в разы, не берём."""
        spec = instance.spec
        total = base + grown
        if not 0.5 * spec.vram_gb <= total <= 2 * spec.vram_gb + 2:
            log.warning("%s: на GPU%s память выросла на %.1f ГБ, а ждали около %.1f — замер не беру. "
                        "Возможно, на карте менялось что-то чужое или скрипт сам выбирает карту",
                        instance.spec.name, gpu.id, grown, spec.vram_gb - base)
            return
        if spec.name not in self.footprints and abs(total - spec.vram_gb) > 0.3:
            log.warning("%s на деле занимает %.1f ГБ, а vram_gb в конфиге %.1f. Считаю по замеру",
                        instance.spec.name, total, spec.vram_gb)
        instance.awake_gb = self.footprints[spec.name] = total

    def note_tail(self, instance: Instance, freed: float) -> None:
        """Запоминает, сколько копия держит во сне, по тому, сколько она отдала."""
        spec = instance.spec
        if instance.awake_gb is None:
            if freed > 0.3 * spec.vram_gb:
                instance.awake_gb = self.footprints[spec.name] = freed + instance.tail
            return
        tail = instance.awake_gb - freed
        if not 0 <= tail <= 0.5 * instance.awake_gb:
            return
        if spec.name not in self.tails and abs(tail - self.settings.asleep_tail_gb) > 0.3:
            log.warning("%s во сне держит %.1f ГБ, а asleep_tail_gb в конфиге %.1f. Считаю по "
                        "замеру, но размещение при старте решается по конфигу — поправьте его",
                        instance.spec.name, tail, self.settings.asleep_tail_gb)
        instance.tail_gb = self.tails[spec.name] = tail

    # ── освобождение места ─────────────────────────────────────────────────

    async def free_up(self, gpu: Gpu, spec: ModelSpec, need: float, deadline: float,
                      wait: bool = True) -> None:
        """Освобождает на карте need ГБ под модель. Вызывать под gpu.lock."""
        await self.refresh()
        if gpu.fits(need):
            return
        victims = await self.claim_victims(gpu, spec, need, deadline, wait)
        try:
            for instance in victims:
                await self.evict(instance)
        finally:
            for instance in victims:
                instance.leaving = False
        await self.settle(gpu, need)

    async def settle(self, gpu: Gpu, need: float) -> None:
        """Ждёт, пока драйвер отдаст память остановленных процессов, и проверяет результат."""
        deadline = time.monotonic() + 10
        await self.refresh(max_age=0)
        while not gpu.fits(need) and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            await self.refresh(max_age=0)
        if not gpu.fits(need):
            raise NoRoom(f"на GPU{gpu.id} после вытеснения {gpu.memory.free_gb:.1f} из нужных "
                         f"{gpu.required(need):.1f} ГБ — проверьте посторонние процессы")

    async def claim_victims(self, gpu: Gpu, spec: ModelSpec, need: float, deadline: float,
                            wait: bool) -> list[Instance]:
        """Выбирает, кого выселить, и закрывает им вход; занятым даёт доработать."""
        claimed: set[Instance] = set()
        announced = False
        try:
            while True:
                now = time.monotonic()
                ready, victims = self.victims(gpu, spec, need, wait)
                claimed = self.reclaim(claimed, victims, now)
                if ready:
                    if announced:
                        log.info("%s: место на GPU%s освободилось", spec.name, gpu.id)
                    return victims
                if now > deadline:
                    raise NoRoom(f"{spec.name}: не дождались места на GPU{gpu.id}, "
                                 f"карту занимали: {self.holding(victims, now)}")
                if not announced:
                    log.info("%s ждёт места на GPU%s: %s дорабатывает запросы или только что "
                             "проснулась", spec.name, gpu.id, self.holding(victims, now))
                    announced = True
                await asyncio.sleep(0.5)
                await self.refresh(max_age=0)
        except BaseException:
            for instance in claimed:
                instance.leaving = False
            raise

    @staticmethod
    def victims(gpu: Gpu, spec: ModelSpec, need: float, wait: bool) -> tuple[bool, list[Instance]]:
        """Кого выселить и можно ли прямо сейчас; NoRoom — если подвинуть вообще некого."""
        if (ready := gpu.evictable(spec, need)) is not None:
            return True, ready
        if not wait:
            raise NoRoom(f"на GPU{gpu.id} сейчас некого выселить без ожидания")
        if (later := gpu.evictable(spec, need, wait=True)) is None:
            raise NoRoom(f"на GPU{gpu.id} нужно {gpu.required(need):.1f} ГБ, свободно "
                         f"{gpu.memory.free_gb:.1f}, и подвинуть некого — проверьте "
                         f"посторонние процессы (nvidia-smi)")
        return False, later

    @staticmethod
    def reclaim(claimed: set[Instance], victims: list[Instance], now: float) -> set[Instance]:
        """Переносит пометку «выселяется» на новый список жертв. Без await: иначе запрос проскочит."""
        fresh = {i for i in victims if not i.resident(now)}
        for instance in claimed - fresh:
            instance.leaving = False
        for instance in fresh:
            instance.leaving = True
        return fresh

    @staticmethod
    def holding(victims: list[Instance], now: float) -> str:
        """Перечисляет тех, из-за кого приходится ждать."""
        return ", ".join(i.spec.name for i in victims if i.busy or i.resident(now))

    async def evict(self, instance: Instance) -> None:
        """Забирает память у копии: усыпляет, а несонливую останавливает."""
        if instance.state is State.AWAKE and instance.can_sleep:
            if await self.put_to_sleep(instance):
                return
        await self.forget(instance)

    async def put_to_sleep(self, instance: Instance) -> bool:
        """Усыпляет копию и замеряет, сколько она отдала; False — если vLLM уснуть не смог."""
        gpu = self.gpus[instance.gpu_id]
        with instance.held():          # вход закрываем до первого await, иначе запрос успеет занять
            if instance.busy:
                return True            # заняли, пока мы сюда шли — усыпим в следующий раз
            try:
                before = await self.free_now(gpu)
                if await instance.set_sleeping(True):
                    self.note_tail(instance, await self.free_now(gpu) - before)
                return True
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                instance.can_sleep = False
                log.warning("%s не засыпает (%s). Больше не укладываю: при нехватке места буду "
                            "останавливать, и следующий запуск будет холодным", instance.spec.name, error)
                return False
            finally:
                self.invalidate()

    async def wake(self, instance: Instance) -> bool:
        """Будит спящую копию и замеряет, сколько она заняла; при неудаче останавливает её."""
        gpu = self.gpus[instance.gpu_id]
        try:
            before = await self.free_now(gpu)
            tail = instance.tail       # база, поверх которой копия доберёт память
            if not await instance.set_sleeping(False):
                return instance.state is State.AWAKE
            self.note_awake(instance, gpu, tail, before - await self.free_now(gpu))
            return True
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            log.warning("%s не проснулась (%s), останавливаю и подниму заново", instance.spec.name, error)
            await self.forget(instance)
            return False
        finally:
            self.invalidate()

    # ── жизненный цикл копий ───────────────────────────────────────────────

    async def launch(self, gpu: Gpu, spec: ModelSpec) -> Instance:
        """Запускает новую копию на карте, где место уже освобождено. Под gpu.lock."""
        instance = self.factory(spec, gpu.id, self.next_port(), self.settings, self.control)
        instance.tail_gb = self.tails.get(spec.name)
        self.instances[spec.name] = instance
        gpu.instances.append(instance)
        before = await self.free_now(gpu)
        try:
            await instance.start()
        except Exception:
            await self.forget(instance)
            raise
        finally:
            self.invalidate()
        if other := (await asyncio.to_thread(instance.devices_seen)) - {str(gpu.id)}:
            log.error("%s: прокси поставила её на GPU%s, а процесс видит CUDA_VISIBLE_DEVICES=%s. "
                      "Уберите жёсткую карту из скрипта, иначе прокси путает, где чья память",
                      instance.spec.name, gpu.id, ", ".join(sorted(other)))
        self.note_awake(instance, gpu, 0.0, before - await self.free_now(gpu))
        return instance

    async def forget(self, instance: Instance) -> None:
        """Останавливает копию и убирает её из учёта."""
        await instance.stop()
        if self.instances.get(instance.spec.name) is instance:
            del self.instances[instance.spec.name]
        if (gpu := self.gpus.get(instance.gpu_id)) and instance in gpu.instances:
            gpu.instances.remove(instance)
        self.invalidate()

    async def reap_dead(self, instance: Instance) -> Instance | None:
        """Вычёркивает копию, чей процесс упал сам; None — если копии больше нет."""
        if instance.alive or (instance.state is State.STARTING and instance.process is None):
            return instance
        if self.instances.get(instance.spec.name) is instance:
            log.warning("%s: процесс умер сам — вычёркиваю", instance.spec.name)
            await self.forget(instance)
        return None

    def next_port(self) -> int:
        """Ищет свободный внутренний порт: сначала с шагом PORT_STEP, потом любой."""
        mine = {instance.port for instance in self.instances.values()}
        ports = self.settings.internal_ports
        for port in [*ports[::PORT_STEP], *ports]:
            if port not in mine and port_is_free(port):
                return port
        raise NoRoom("свободных внутренних портов нет: расширьте internal_ports "
                     "или проверьте процессы с прошлых запусков (ss -tlnp)")

    # ── обучение ───────────────────────────────────────────────────────────

    def usable_gb(self, gpu: Gpu) -> float:
        """Сколько памяти карты достанется обучению, когда прокси уберёт свои модели."""
        return max(0.0, gpu.memory.total_gb - gpu.foreign_gb() - self.settings.reserve_gb)

    def pick_gpus(self, vram_gb: float) -> list[int]:
        """Набирает карты под нужный объём, начиная с тех, что меньше всего мешают моделям."""
        free_choice = sorted((gpu for gpu in self.gpus.values() if not gpu.blocked),
                             key=lambda g: (any(i.spec.priority for i in g.instances),
                                            sum(i.state is State.AWAKE for i in g.instances),
                                            -self.usable_gb(g)))
        chosen: list[int] = []
        total = 0.0
        for gpu in free_choice:
            if total >= vram_gb:
                break
            chosen.append(gpu.id)
            total += self.usable_gb(gpu)
        if total < vram_gb:
            raise ValueError(f"обучению нужно {vram_gb:.1f} ГБ, а свободные карты дают "
                             f"{total:.1f} ГБ ({len(free_choice)} шт). Уменьшите запрос "
                             f"или дождитесь конца другого обучения")
        return sorted(chosen)

    def training_targets(self, gpus: list[int] | None, vram_gb: float | None) -> set[int]:
        """Решает, какие карты отдать обучению: по списку, по объёму или все."""
        if vram_gb is not None:
            if gpus is not None:
                raise ValueError("укажите либо gpus, либо vram_gb, но не оба сразу")
            if vram_gb <= 0:
                raise ValueError("vram_gb должно быть больше нуля")
            return set(self.pick_gpus(vram_gb))
        if gpus is None:
            return set(self.gpus)
        if not gpus:
            raise ValueError("gpus пуст: перечислите карты, попросите объём через vram_gb "
                             "или уберите поле совсем, чтобы забрать все карты")
        if unknown := set(gpus) - set(self.gpus):
            raise ValueError(f"карт {sorted(unknown)} нет, есть {sorted(self.gpus)}")
        return set(gpus)

    async def start_training(self, gpus: list[int] | None, owner: str, pid: int | None,
                             vram_gb: float | None = None) -> Lease:
        """Отдаёт карты обучению: закрывает вход, гасит модели и возвращает аренду."""
        await self.refresh()
        targets = self.training_targets(gpus, vram_gb)
        stat = proc_stat(pid) if pid is not None else None
        if pid is not None and stat is None:
            raise ValueError(f"процесса {pid} нет — проверьте pid")
        lease = Lease(uuid.uuid4().hex[:8], targets, owner, pid, stat[1] if stat else None)
        self.leases[lease.id] = lease
        for gpu_id in targets:
            self.gpus[gpu_id].blocked.add(lease.id)
            for instance in self.gpus[gpu_id].instances:
                instance.leaving = True
        log.warning("обучение %s (%s) забирает GPU %s", lease.id, owner, sorted(targets))
        try:
            # return_exceptions: иначе первая ошибка вернёт управление, пока остальные карты дренируются
            done = await asyncio.gather(*(self.vacate(self.gpus[g]) for g in sorted(targets)),
                                        return_exceptions=True)
        except BaseException:
            await self.end_training(lease.id)
            raise
        if failed := [item for item in done if isinstance(item, BaseException)]:
            await self.end_training(lease.id)
            raise failed[0]
        return lease

    async def vacate(self, gpu: Gpu) -> None:
        """Гасит все модели на карте, дав запросам доработать не дольше drain_timeout."""
        async with gpu.lock:
            for instance in gpu.instances:
                instance.leaving = True
            deadline = time.monotonic() + self.settings.drain_timeout
            while any(i.busy for i in gpu.instances) and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
            if busy := [i.spec.name for i in gpu.instances if i.busy]:
                log.warning("GPU%s: %s не доработали за %.0f сек, гашу вместе с запросами",
                            gpu.id, ", ".join(busy), self.settings.drain_timeout)
            for instance in list(gpu.instances):
                await self.forget(instance)
            await self.refresh(max_age=0)

    async def end_training(self, lease_id: str) -> Lease:
        """Снимает аренду и возвращает карты моделям."""
        lease = self.leases.pop(lease_id)
        for gpu_id in lease.gpus:
            if not (gpu := self.gpus.get(gpu_id)):
                continue
            gpu.blocked.discard(lease_id)
            if not gpu.blocked and not gpu.lock.locked():
                for instance in gpu.instances:
                    instance.leaving = False
        log.warning("обучение %s (%s) закончилось, GPU %s снова доступны моделям",
                    lease.id, lease.owner, sorted(lease.gpus))
        self.invalidate()
        return lease

    async def check_leases(self) -> None:
        """Снимает аренды, чей процесс обучения уже завершился."""
        for lease in list(self.leases.values()):
            if not lease.alive():
                log.warning("процесс обучения %s (pid %s) завершился, а аренду не сняли — снимаю сам",
                            lease.id, lease.pid)
                await self.end_training(lease.id)

    # ── фоновое обслуживание ───────────────────────────────────────────────

    async def preload(self) -> None:
        """Поднимает модели при старте: приоритетные первыми, они же в конце ещё раз."""
        specs = sorted(self.settings.models.values(), key=lambda s: not s.priority)
        for spec in specs + [s for s in specs if s.priority]:
            try:
                (await self.acquire(spec.name)).release()
                log.info("%s готова", spec.name)
            except Exception as error:
                log.error("%s не поднялась: %s", spec.name, error)

    async def housekeeping(self) -> None:
        """Раз в 15 секунд наводит порядок; задача не должна умирать ни при какой ошибке."""
        while True:
            await asyncio.sleep(15)
            try:
                await self.tidy()
            except Exception as error:
                log.warning("уборка сорвалась: %s", error)

    async def tidy(self) -> None:
        """Снимает брошенные аренды, гасит лишнее и поднимает приоритетные."""
        for step in (self.check_leases, self.reap, self.restore_priority):
            try:
                await step()
            except Exception as error:
                log.warning("уборка (%s) не удалась: %s", step.__name__, error)

    async def reap(self) -> None:
        """Обходит простаивающие копии; карты, где идёт переключение, не трогает."""
        for instance in sorted(self.instances.values(),
                               key=lambda i: (i.state is State.AWAKE, i.last_used)):
            if await self.reap_dead(instance) is None or instance.busy or instance.leaving:
                continue
            gpu = self.gpus.get(instance.gpu_id)
            if gpu is None or gpu.lock.locked():
                continue
            try:
                async with gpu.lock:
                    await self.reap_one(instance, gpu)
            except Exception as error:
                log.warning("%s: уборка не удалась: %s", instance.spec.name, error)

    async def reap_one(self, instance: Instance, gpu: Gpu) -> None:
        """Решает судьбу одной простаивающей копии. Под gpu.lock."""
        if (instance.busy or instance.leaving or instance.spec.priority
                or instance not in gpu.instances):
            return
        idle = instance.idle_for
        if (instance.state is State.AWAKE and instance.can_sleep
              and idle > self.settings.idle_sleep_sec):
            log.info("%s простаивает %.0f сек — усыпляю", instance.spec.name, idle)
            await self.put_to_sleep(instance)
        elif idle > self.settings.idle_stop_sec and (instance.state is State.ASLEEP
                                                     or not instance.can_sleep):
            log.info("%s не нужна уже %.0f сек — останавливаю", instance.spec.name, idle)
            await self.forget(instance)

    async def restore_priority(self) -> None:
        """Держит приоритетные модели наготове, пока никто не ждёт ответа."""
        if any(instance.busy for instance in self.instances.values()):
            return
        for spec in self.settings.models.values():
            instance = self.instances.get(spec.name)
            if not spec.priority or (instance is not None and instance.state is not State.ASLEEP):
                continue
            await self.refresh()
            if instance is not None:
                await self.revive(self.gpus[instance.gpu_id], spec, instance)
            else:
                await self.revive(self.roomiest(self.need_for(spec, None)), spec, None)

    def roomiest(self, need: float) -> Gpu | None:
        """Самая свободная карта, куда модель встанет без выселения соседей."""
        free = [g for g in self.gpus.values()
                if not g.blocked and not g.lock.locked() and g.fits(need)]
        return max(free, key=lambda g: g.memory.free_gb) if free else None

    async def revive(self, gpu: Gpu | None, spec: ModelSpec, sleeper: Instance | None) -> None:
        """Поднимает приоритетную модель на карте, если там есть место без выселения."""
        if gpu is None or gpu.blocked or gpu.lock.locked():
            return
        if sleeper is not None and not gpu.fits(sleeper.wake_need):
            return
        async with gpu.lock:
            log.info("на GPU%s есть место — %s приоритетную %s", gpu.id,
                     "бужу" if sleeper else "запускаю", spec.name)
            await (self.wake(sleeper) if sleeper else self.launch(gpu, spec))

    # ── отчёт и остановка ──────────────────────────────────────────────────

    async def report(self) -> dict:
        """Состояние карт, аренд и копий для /health и стартового лога."""
        try:
            await self.refresh()
        except NoRoom as error:
            return {"error": str(error)}
        return {"gpus": [repr(gpu) for gpu in self.gpus.values()],
                "training": [lease.describe() for lease in self.leases.values()],
                "models": {name: self.describe(self.instances.get(name))
                           for name in self.settings.models}}

    @staticmethod
    def describe(instance: Instance | None) -> dict | None:
        """Состояние одной копии для /health."""
        if instance is None:
            return None
        return {"state": instance.state.value, "gpu": instance.gpu_id, "port": instance.port,
                "busy": instance.busy, "idle_sec": round(instance.idle_for),
                "memory_gb": round(instance.holds_gb, 2),
                "measured": instance.awake_gb is not None}

    async def shutdown(self) -> None:
        """Гасит все модели разом и закрывает HTTP-клиент, если он наш."""
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*(i.stop() for i in self.instances.values()), return_exceptions=True)
        if self._owns_control:
            await self.control.aclose()


# ── HTTP ───────────────────────────────────────────────────────────────────

class QuietServer(uvicorn.Server):
    """uvicorn без своей обработки сигналов: иначе после SIGTERM процессы vLLM остаются на картах."""

    def install_signal_handlers(self) -> None:      # uvicorn старше 0.29
        pass

    @contextlib.contextmanager
    def capture_signals(self):                      # uvicorn 0.29 и новее
        yield


def training_request(body: dict) -> tuple[list[int] | None, str, int | None, float | None]:
    """Разбирает тело запроса на аренду карт."""
    if unknown := sorted(set(body) - TRAINING_KEYS):
        raise HTTPException(400, f"непонятные поля {', '.join(unknown)}: "
                                 f"допустимы {', '.join(sorted(TRAINING_KEYS))}")
    gpus, pid, need = body.get("gpus"), body.get("pid"), body.get("vram_gb")
    if gpus is not None and not (isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)):
        raise HTTPException(400, "gpus должен быть списком номеров карт или null")
    if pid is not None and not isinstance(pid, int):
        raise HTTPException(400, "pid должен быть числом или null")
    if need is not None and (isinstance(need, bool) or not isinstance(need, (int, float))):
        raise HTTPException(400, "vram_gb должен быть числом ГБ или null")
    return gpus, str(body.get("owner") or "обучение"), pid, float(need) if need else None


def usage_of(response: httpx.Response) -> dict:
    """Достаёт usage из ответа; битое тело не должно ронять запрос."""
    with contextlib.suppress(ValueError, AttributeError):
        return response.json().get("usage") or {}
    return {}


class Stream:
    """Перекачивает стрим vLLM клиенту: upstream дочитываем всегда, медленного клиента обрываем."""

    def __init__(self, upstream: httpx.Response, instance: Instance, tag: str, started: float):
        self.upstream = upstream
        self.instance = instance
        self.tag = tag
        self.started = started
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.buffered = 0

    async def pump(self) -> None:
        """Читает ответ модели в очередь и отпускает модель, когда он дописан."""
        dropped = False
        try:
            async for chunk in self.upstream.aiter_bytes():
                if dropped:
                    continue      # клиента нет, но upstream дочитываем, чтобы не рвать генерацию
                if self.buffered + len(chunk) > STREAM_BUFFER:
                    dropped = True
                    log.warning("%s клиент не забирает стрим (%.1f МБ) — обрываю его, "
                                "модели даю дописать", self.tag, self.buffered / 2 ** 20)
                    continue
                self.buffered += len(chunk)
                self.queue.put_nowait(chunk)
            log.info("%s стрим за %.1f сек", self.tag, time.monotonic() - self.started)
        except Exception as error:
            log.error("%s стрим оборвался: %s", self.tag, error)
        finally:
            self.instance.release()
            self.queue.put_nowait(None)
            await self.upstream.aclose()

    async def chunks(self):
        """Отдаёт клиенту накопленные куски."""
        while (chunk := await self.queue.get()) is not None:
            self.buffered -= len(chunk)
            yield chunk


class Proxy:
    """Принимает запросы на внешних портах и пересылает их нужной модели."""

    def __init__(self, settings: Settings, cluster: Cluster):
        self.settings = settings
        self.cluster = cluster
        self.counter = itertools.count(1)
        # лимит соединений снят, чтобы vLLM собирал большой батч; таймаут чтения из конфига
        self.inference = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.read_timeout, connect=10),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))

    # ── приложения ─────────────────────────────────────────────────────────

    def application(self, port: int) -> FastAPI:
        """Собирает приложение для одного внешнего порта."""
        app = FastAPI()
        created = int(time.time())

        @app.get("/v1/models")
        async def models(request: Request):
            """Список моделей порта: отвечаем из конфига, чтобы не будить их ради списка."""
            self.check_key(request)
            names = [alias for name in self.settings.on_port(port)
                     for alias in (name, *self.settings.models[name].aliases)]
            return {"object": "list", "data": [{"id": name, "object": "model", "created": created,
                                                "owned_by": "vllm"} for name in names]}

        @app.post("/v1/{path:path}")
        async def handle(path: str, request: Request):
            """Пересылает запрос модели; пути с точками и процентами не пускаем."""
            self.check_key(request)
            if not SAFE_PATH.fullmatch(path):
                raise HTTPException(404, "такого эндпоинта нет")
            return await self.forward(port, path, request)

        return app

    def admin_application(self) -> FastAPI:
        """Собирает служебное приложение для 127.0.0.1."""
        app = FastAPI()
        app.get("/health")(self.cluster.report)
        app.get("/admin/training")(self.show_leases)
        app.post("/admin/training")(self.open_lease)
        app.delete("/admin/training/{lease_id}")(self.close_lease)
        return app

    async def show_leases(self) -> dict:
        """Показывает текущие аренды под обучение."""
        return {"leases": [lease.describe() for lease in self.cluster.leases.values()]}

    async def open_lease(self, request: Request) -> dict:
        """Отдаёт карты обучению; отвечает, когда модели на них уже погашены."""
        raw = await request.body()
        gpus, owner, pid, need = training_request(await self.parse(raw) if raw.strip() else {})
        try:
            lease = await self.cluster.start_training(gpus, owner, pid, need)
        except ValueError as error:
            raise HTTPException(400, str(error))
        return {**lease.describe(),
                "free_gb": round(sum(self.cluster.gpus[g].memory.free_gb for g in lease.gpus), 1),
                "memory": [repr(self.cluster.gpus[g]) for g in sorted(lease.gpus)]}

    async def close_lease(self, lease_id: str) -> dict:
        """Возвращает карты моделям."""
        if lease_id not in self.cluster.leases:
            raise HTTPException(404, f"аренды {lease_id} нет")
        return (await self.cluster.end_training(lease_id)).describe()

    def check_key(self, request: Request) -> None:
        """Проверяет Authorization, если в конфиге задан api_key."""
        key = self.settings.api_key
        if key and not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                                           f"Bearer {key}".encode()):
            raise HTTPException(401, "нужен заголовок Authorization: Bearer <api_key>")

    # ── обработка запроса ──────────────────────────────────────────────────

    async def forward(self, port: int, path: str, request: Request) -> Response:
        """Определяет модель по телу запроса и отправляет его ей."""
        tag = f"[#{next(self.counter)}]"
        started = time.monotonic()
        raw = await request.body()
        body = await self.parse(raw)
        name = self.settings.resolve(body.get("model"), port)
        if name is None:
            raise HTTPException(400, f"на порту {port} доступны: "
                                     f"{', '.join(self.settings.on_port(port))}")
        log.info("%s :%s /v1/%s -> %s", tag, port, path, name)
        return await self.send(name, path, request.url.query, raw, body, tag, started)

    @staticmethod
    async def parse(raw: bytes) -> dict:
        """Разбирает тело запроса и проверяет, что это JSON-объект."""
        try:
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "тело запроса должно быть корректным JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "ожидается JSON-объект")
        return body

    async def send(self, name: str, path: str, query: str, raw: bytes,
                   body: dict, tag: str, started: float) -> Response:
        """Занимает модель, пересылает ей запрос и возвращает ответ или стрим."""
        instance = upstream = None
        try:
            instance = await self.cluster.acquire(name)
            if (waited := time.monotonic() - started) > 1:
                log.info("%s место получено за %.1f сек", tag, waited)
            upstream = await self.inference.send(
                self.build(instance, name, path, query, raw, body), stream=True)
            if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                response = self.relay(upstream, instance, tag, started)
                instance = upstream = None      # дальше модель и соединение отпустит relay
                return response
            await upstream.aread()
            return self.finish(upstream, tag, started)
        except NoRoom as error:
            raise self.no_room(tag, error)
        except (httpx.HTTPError, RuntimeError, OSError, ValueError) as error:
            raise self.upstream_failed(tag, name, instance, error)
        finally:
            if instance is not None:
                instance.release()
            if upstream is not None:
                await upstream.aclose()

    @staticmethod
    def no_room(tag: str, error: NoRoom) -> HTTPException:
        """Превращает нехватку места в 503 с подсказкой, когда повторить."""
        log.error("%s %s", tag, error)
        retry = "300" if isinstance(error, TrainingInProgress) else "30"
        return HTTPException(503, str(error), headers={"Retry-After": retry})

    def upstream_failed(self, tag: str, name: str, instance: Instance | None,
                        error: Exception) -> HTTPException:
        """Превращает сбой модели в 502 и проверяет, не умерла ли копия."""
        log.error("%s %s не отработала: %s", tag, name, error)
        if instance is not None:      # пусть следующий запрос не бьётся о труп
            self.cluster.spawn(self.cluster.reap_dead(instance))
        return HTTPException(502, f"{name}: {error}")

    def build(self, instance: Instance, name: str, path: str, query: str,
              raw: bytes, body: dict) -> httpx.Request:
        """Собирает запрос к vLLM; алиас в поле model заменяем на имя, под которым он её знает."""
        content = raw if body.get("model") == name else \
            json.dumps({**body, "model": name}, ensure_ascii=False).encode()
        return self.inference.build_request(
            "POST", f"{instance.url}/v1/{path}" + (f"?{query}" if query else ""),
            content=content, headers={"content-type": "application/json"})

    def relay(self, upstream: httpx.Response, instance: Instance,
              tag: str, started: float) -> StreamingResponse:
        """Отдаёт стрим по мере генерации; модель отпустит Stream, когда vLLM допишет ответ."""
        stream = Stream(upstream, instance, tag, started)
        self.cluster.spawn(stream.pump())
        # без этих заголовков nginx копит стрим у себя и отдаёт клиенту одним куском
        return StreamingResponse(stream.chunks(), upstream.status_code,
                                 media_type=upstream.headers["content-type"],
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    @staticmethod
    def finish(response: httpx.Response, tag: str, started: float) -> Response:
        """Отдаёт ответ vLLM без изменений и пишет в лог число токенов."""
        content_type = response.headers.get("content-type", "")
        if response.status_code != 200:
            log.warning("%s модель вернула %s: %s", tag, response.status_code, response.text[:300])
        elif "application/json" not in content_type:
            log.warning("%s не-JSON ответ", tag)
        elif log.isEnabledFor(logging.INFO):    # разбирать тело стоит только ради строчки в логе
            usage = usage_of(response)
            log.info("%s ответ за %.1f сек, токенов: %s + %s", tag, time.monotonic() - started,
                     usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"))
        return Response(response.content, response.status_code, media_type=content_type)

    async def run(self, stop: asyncio.Event) -> None:
        """Поднимает внешние порты и служебный порт и работает до остановки."""
        configs = [uvicorn.Config(self.application(port), host="0.0.0.0", port=port,
                                  log_level="warning") for port in self.settings.listen]
        configs.append(uvicorn.Config(self.admin_application(), host="127.0.0.1",
                                      port=self.settings.admin_port, log_level="warning"))
        servers = [QuietServer(config) for config in configs]
        serving = [asyncio.create_task(server.serve()) for server in servers]
        stopping = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait([*serving, stopping], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for server in servers:
                server.should_exit = True
            await asyncio.gather(*serving, return_exceptions=True)
            stopping.cancel()
            await self.inference.aclose()


# ── запуск ─────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, keep: int) -> Path:
    """Настраивает логи: в файл подробно, в консоль только предупреждения."""
    path = new_log(log_dir, "proxy", keep)
    to_file = logging.handlers.RotatingFileHandler(path, maxBytes=50 * 2 ** 20, backupCount=3,
                                                   encoding="utf-8")
    to_file.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                           datefmt="%H:%M:%S"))
    to_console = logging.StreamHandler()
    to_console.setLevel(os.getenv("CONSOLE_LEVEL", "WARNING"))
    to_console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.DEBUG, handlers=[to_file, to_console])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return path


async def announce(settings: Settings, cluster: Cluster, log_path: Path, killed: list[str]) -> None:
    """Печатает при старте раскладку портов, состояние карт и предупреждения."""
    state = await cluster.report()
    lines = [f"лог: {log_path}"]
    lines += [f"порт :{port} -> {', '.join(settings.on_port(port))}" for port in settings.listen]
    lines.append(f"служебный порт: http://127.0.0.1:{settings.admin_port} (/health, /admin/training)")
    lines += state.get("gpus", [])
    if error := state.get("error"):
        lines.append(f"ВНИМАНИЕ: карты не опрошены — {error}")
    if killed:
        lines.append("погашены процессы от прошлого запуска: " + ", ".join(killed))
    lines.append("модели поднимаются при запуске" if settings.preload
                 else "модели поднимаются по первому запросу")
    if not settings.api_key:
        lines.append("ВНИМАНИЕ: api_key не задан, а внешние порты слушают 0.0.0.0 — "
                     "модели доступны любому, кто достанет до машины по сети")
    print("\n".join(lines), flush=True)
    log.info("%s", "\n".join(lines))

    if error:
        log.error("карты не опрошены: %s", error)
    if missing := [name for name, spec in settings.models.items()
                   if "HF_HOME" not in {**os.environ, **settings.env, **spec.env}]:
        log.warning("HF_HOME не задан для %s — vLLM может заново скачивать чекпоинты",
                    ", ".join(missing))


def check_ports(settings: Settings) -> None:
    """Проверяет, что внешние и служебный порты свободны."""
    busy = [port for port in settings.listen if not port_is_free(port)]
    if not port_is_free(settings.admin_port, "127.0.0.1"):
        busy.append(settings.admin_port)
    if busy:
        fail(f"порты {busy} уже заняты, прокси уже запущена?")


def install_signals(stop: asyncio.Event) -> None:
    """Вешает обработчики: первый сигнал просит остановиться, второй выходит немедленно."""
    def on_signal() -> None:
        if stop.is_set():          # ждать не хотят; хвосты уберёт следующий запуск
            log.warning("повторный сигнал, выхожу не дожидаясь остановки моделей")
            os._exit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) != signal.SIG_IGN:   # под nohup сигнал велели не замечать
            loop.add_signal_handler(sig, on_signal)


async def main() -> None:
    """Загружает конфиг, поднимает кластер и HTTP и корректно всё гасит по сигналу."""
    settings = Settings.load(os.getenv("CONFIG", HERE / "config.yaml"))
    path = setup_logging(Path(os.getenv("LOG_DIR", HERE / "logs")), settings.log_keep)
    check_ports(settings)
    killed = await kill_orphans(settings)
    cluster = Cluster(settings)

    stop = asyncio.Event()
    install_signals(stop)
    await announce(settings, cluster, path, killed)

    background = [asyncio.create_task(cluster.housekeeping())]
    if settings.preload:
        background.append(asyncio.create_task(cluster.preload()))
    try:
        await Proxy(settings, cluster).run(stop)
    finally:
        for task in background:
            task.cancel()
        await cluster.shutdown()
        log.info("прокси остановлена")


def run() -> None:
    """Точка входа; использует uvloop, если он установлен."""
    try:
        import uvloop
        runner = uvloop.run
    except ImportError:
        runner = asyncio.run
    with contextlib.suppress(KeyboardInterrupt):
        runner(main())


if __name__ == "__main__":
    run()
