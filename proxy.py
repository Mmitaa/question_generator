"""Прокси перед несколькими vLLM. Сама усыпляет, будит, переносит и масштабирует модели между картами, а на время обучения отдаёт карты ему. Запускать через python proxy.py"""

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
import shlex
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

HERE = Path(__file__).parent
log = logging.getLogger("proxy")

SAFE_PATH = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")   # только обычные сегменты пути, без точек и процентов
BIG_BODY = 1_000_000                                             # тела больше мегабайта гоняем через JSON в отдельном потоке
PORT_STEP = 10         # vLLM занимает под свои соединения порты сразу за HTTP-портом, поэтому инстансам даём порты с запасом
RESIDENCY_GRACE = 3.0  # столько секунд без запросов, и только что проснувшуюся модель уже не защищаем, ею никто не пользуется
STREAM_BUFFER = 32 * 2 ** 20   # столько байт стрима держим для медленного клиента, дальше обрываем его, чтобы не копить ответ в памяти

# поля, которые понимаем в секции модели, всё остальное считаем опечаткой
MODEL_KEYS = frozenset({"cmd", "cwd", "script", "vram_gb", "port", "aliases",
                        "priority", "replicas", "env", "log_dir"})
# прокси ставит модель на одну карту, поэтому многокарточные запуски ловим сразу в конфиге
PARALLEL_FLAGS = frozenset({"--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp",
                            "--data-parallel-size", "-dp"})


class NoRoom(RuntimeError):
    """Кидаем, когда модель не влезает ни на одну карту даже после вытеснения соседей."""


class TrainingInProgress(NoRoom):
    """Кидаем, когда все карты отданы под обучение."""


# ── конфигурация ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """Описание модели из конфига. Как запускать, сколько памяти нужно, на каком порту слушать и сколько копий можно держать."""

    name: str
    cmd: list[str]
    cwd: Path
    vram_gb: float
    port: int
    script: Path | None = None
    log_dir: Path | None = None
    aliases: tuple[str, ...] = ()
    priority: bool = False
    replicas: int = 1
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, raw: dict) -> ModelSpec:
        """Собирает ModelSpec из секции конфига и сразу проверяет его, при ошибке останавливает запуск с понятным сообщением."""
        if not isinstance(raw, dict):
            raise SystemExit(f"{name}: секция модели должна быть словарём, а не {type(raw).__name__}")
        if unknown := sorted(set(raw) - MODEL_KEYS):
            raise SystemExit(f"{name}: непонятные поля {', '.join(unknown)}. "
                             f"Допустимы: {', '.join(sorted(MODEL_KEYS))}")
        for required in ("cwd", "vram_gb", "port"):
            if required not in raw:
                raise SystemExit(f"{name}: в конфиге нет обязательного поля {required}")

        script = Path(raw["script"]) if raw.get("script") else None
        if bool(script) == bool(raw.get("cmd")):
            raise SystemExit(f"{name}: укажите либо script, либо cmd, но не оба")
        if script and not script.exists():
            raise SystemExit(f"{name}: скрипта {script} не существует")

        try:
            spec = cls(name=name, cmd=shlex.split(raw.get("cmd", "")), cwd=Path(raw["cwd"]),
                       script=script, vram_gb=float(raw["vram_gb"]), port=int(raw["port"]),
                       aliases=tuple(raw.get("aliases", ())),
                       priority=bool(raw.get("priority")),
                       replicas=int(raw.get("replicas", 1)),
                       env={k: str(v) for k, v in (raw.get("env") or {}).items()},
                       log_dir=Path(raw["log_dir"]) if raw.get("log_dir") else None)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"{name}: в конфиге неверное значение — {error}")

        if not spec.cwd.is_dir():
            raise SystemExit(f"{name}: каталог {spec.cwd} не существует")
        if spec.replicas < 1:
            raise SystemExit(f"{name}: replicas должно быть не меньше 1")
        if spec.vram_gb <= 0:
            raise SystemExit(f"{name}: vram_gb должно быть больше нуля")
        for flag, value in itertools.zip_longest(spec.cmd, spec.cmd[1:], fillvalue=""):
            head, _, inline = flag.partition("=")
            if head in PARALLEL_FLAGS and (inline or value) not in ("", "1"):
                raise SystemExit(f"{name}: {head} {inline or value} — прокси ставит модель на одну "
                                 f"карту (CUDA_VISIBLE_DEVICES с одним номером) и несколько карт на "
                                 f"инстанс не умеет. Уберите флаг или запускайте такую модель мимо прокси")
        has_pyproject = any((folder / "pyproject.toml").exists()
                            for folder in (spec.cwd, *spec.cwd.parents))
        if spec.cmd[:1] == ["poetry"] and not has_pyproject:
            raise SystemExit(f"{name}: ни в {spec.cwd}, ни выше нет pyproject.toml — "
                             f"poetry run там не заработает. Укажите в cwd каталог "
                             f"проекта или вызывайте бинарник из venv напрямую")
        return spec


@dataclass(frozen=True)
class Settings:
    """Весь конфиг прокси. Создаётся через Settings.load()."""

    models: dict[str, ModelSpec]
    internal_ports: range
    env: dict[str, str] = field(default_factory=dict)
    admin_port: int = 4999
    api_key: str | None = None
    reserve_gb: float = 1.0
    asleep_tail_gb: float = 0.5
    sleep_level: int = 1
    idle_sleep_sec: float = 120
    idle_stop_sec: float = 3600
    min_residency_sec: float = 30
    start_timeout: float = 900
    switch_timeout: float = 120
    queue_timeout: float = 180
    drain_timeout: float = 600
    read_timeout: float | None = None
    scale_up_busy: int = 8
    log_keep: int = 20
    preload: bool = True
    vllm_log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Проверяет конфиг на противоречия и строит словари для быстрого поиска модели."""
        if not self.models:
            raise SystemExit("в конфиге нет ни одной модели")
        if not self.internal_ports:
            raise SystemExit("internal_ports пуст: укажите [низ, верх], где верх не меньше низа")
        if self.sleep_level not in (1, 2):
            raise SystemExit(f"sleep_level должен быть 1 или 2, а не {self.sleep_level}")
        if self.reserve_gb < 0 or self.asleep_tail_gb < 0:
            raise SystemExit("reserve_gb и asleep_tail_gb не могут быть отрицательными")

        by_port: dict[int, list[str]] = {}
        by_alias: dict[str, str] = {}
        for name, spec in self.models.items():
            if spec.port in self.internal_ports:
                raise SystemExit(f"{name}: внешний порт {spec.port} попал в internal_ports")
            if spec.port == self.admin_port:
                raise SystemExit(f"{name}: внешний порт {spec.port} совпадает с admin_port")
            by_port.setdefault(spec.port, []).append(name)
            for alias in (name, *spec.aliases):
                if alias in by_alias:
                    raise SystemExit(f"имя {alias!r} занято двумя моделями: "
                                     f"{by_alias[alias]} и {name}")
                by_alias[alias] = name
        if self.admin_port in self.internal_ports:
            raise SystemExit(f"admin_port {self.admin_port} попал в internal_ports")
        if self.min_residency_sec >= self.queue_timeout:
            raise SystemExit("min_residency_sec должен быть меньше queue_timeout, "
                             "иначе ждущие запросы не дождутся карты")
        copies = sum(spec.replicas for spec in self.models.values())
        if len(self.internal_ports) < copies:
            raise SystemExit(f"internal_ports вмещает {len(self.internal_ports)} портов, "
                             f"а копий моделей может быть до {copies} — расширьте диапазон")
        # dataclass заморожен, поэтому пишем в обход через object.__setattr__
        object.__setattr__(self, "_by_port", by_port)
        object.__setattr__(self, "_by_alias", by_alias)
        object.__setattr__(self, "vllm_log_level", self.vllm_log_level.upper())

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        """Читает config.yaml и собирает из него Settings. Непонятные и недостающие поля разбирает здесь же, чтобы ошибка конфига не превращалась в traceback."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{path}: ожидается словарь с настройками")
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - known):
            raise SystemExit(f"{path}: непонятные поля {', '.join(unknown)}. "
                             f"Допустимы: {', '.join(sorted(known))}")
        for required in ("models", "internal_ports"):
            if required not in data:
                raise SystemExit(f"{path}: в конфиге нет обязательного поля {required}")
        try:
            low, high = (int(edge) for edge in data["internal_ports"])
        except (TypeError, ValueError):
            raise SystemExit(f"{path}: internal_ports должен быть парой чисел [низ, верх]")
        if not isinstance(data["models"], dict):
            raise SystemExit(f"{path}: models должен быть словарём «имя: настройки»")
        skip = {"models", "internal_ports", "env"}
        scalars = {f.name: data[f.name] for f in fields(cls)
                   if f.name in data and f.name not in skip}
        return cls(models={name: ModelSpec.from_config(name, raw)
                           for name, raw in data["models"].items()},
                   internal_ports=range(low, high + 1),
                   env={k: str(v) for k, v in (data.get("env") or {}).items()},
                   **scalars)

    @property
    def listen(self) -> list[int]:
        """Внешние порты, на которых поднимаем HTTP."""
        return sorted(self._by_port)

    @property
    def wait_budget(self) -> float:
        """Сколько всего запрос может простоять в очереди к модели: ожидание места плюс один холодный старт, который в это время делает чужой запрос."""
        return self.queue_timeout + self.start_timeout

    def on_port(self, port: int) -> list[str]:
        """Возвращает модели, закреплённые за портом."""
        return self._by_port.get(port, [])

    def resolve(self, requested: str | None, port: int) -> str | None:
        """Определяет, какую модель хотел клиент. Если на порту одна модель, поле model в запросе не важно."""
        here = self.on_port(port)
        if len(here) == 1:
            return here[0]
        name = self._by_alias.get(requested) if isinstance(requested, str) else None
        return name if name in here else None


def strip_venv_from_path(path: str, virtual_env: str | None) -> str:
    """Убирает из PATH venv самого прокси, иначе poetry в инстансах возьмёт его вместо проектного."""
    if not virtual_env:
        return path
    unwanted = str(Path(virtual_env) / "bin")
    return os.pathsep.join(part for part in path.split(os.pathsep) if part != unwanted)


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """Проверяет, свободен ли порт, пробуя занять его на том же адресе, на котором будем слушать. Без SO_REUSEADDR, иначе чужой сокет в TIME_WAIT покажется свободным портом."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def new_log(folder: Path, prefix: str, keep: int = 20) -> Path:
    """Возвращает путь для нового лога с датой в имени, переводит на него симлинк latest и удаляет старые логи сверх keep."""
    folder.mkdir(parents=True, exist_ok=True)
    old = sorted(folder.glob(f"{prefix}_20*"))
    for stale in old[:max(0, len(old) - keep + 1)]:
        with contextlib.suppress(OSError):
            stale.unlink()
    path = folder / f"{prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    link = folder / f"{prefix}_latest.log"
    with contextlib.suppress(OSError):
        link.unlink(missing_ok=True)
        link.symlink_to(path.name)
    return path


def encode_json(body: dict) -> bytes:
    """Собирает тело запроса обратно в JSON."""
    return json.dumps(body, ensure_ascii=False).encode()


async def off_loop(func: Callable[..., Any], *args: Any, size: int) -> Any:
    """Большие тела гоняем через JSON в отдельном потоке, чтобы не держать event loop."""
    if size > BIG_BODY:
        return await asyncio.to_thread(func, *args)
    return func(*args)


async def wait_unlocked(lock: asyncio.Lock, deadline: float, what: str) -> None:
    """Ждёт, пока освободится lock карты, но не дольше deadline. Сам lock не берёт."""
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

# эти переменные остались от окружения прокси и инстансам их передавать нельзя
INHERITED_VENV = ("VIRTUAL_ENV", "POETRY_ACTIVE", "PYTHONHOME", "PYTHONPATH")

GIB = 1024 ** 3  # байт в гибибайте


async def read_gpu_memory() -> dict[int, Memory]:
    """Узнаёт свободную память на всех картах. Сначала пробует pynvml, если он не установлен или не работает, один раз предупреждает и дальше вызывает nvidia-smi."""
    if not getattr(read_gpu_memory, "nvml_broken", False):
        try:
            return await asyncio.to_thread(_nvml_memory)
        except Exception as error:
            read_gpu_memory.nvml_broken = True
            log.warning("pynvml недоступен (%s), читаю память через nvidia-smi. "
                        "Поставьте nvidia-ml-py в окружение прокси", error)
    return await _smi_memory()


def _nvml_memory() -> dict[int, Memory]:
    """Читает память карт через pynvml."""
    import pynvml

    if not getattr(_nvml_memory, "ready", False):
        pynvml.nvmlInit()
        _nvml_memory.ready = True
    result = {}
    for index in range(pynvml.nvmlDeviceGetCount()):
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
        result[index] = Memory(info.total / GIB, info.free / GIB)
    return result


async def _smi_memory() -> dict[int, Memory]:
    """Читает память карт через nvidia-smi. Если и он недоступен, дальше работать вслепую нельзя."""
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await process.communicate()
        ok = process.returncode == 0
    except OSError:
        ok = False
    if not ok:
        raise NoRoom("ни pynvml, ни nvidia-smi недоступны — размещать вслепую нельзя")
    try:
        rows = [list(map(int, line.split(","))) for line in out.decode().strip().splitlines()]
        return {index: Memory(total / 1024, free / 1024) for index, total, free in rows}
    except ValueError as error:
        raise NoRoom(f"nvidia-smi ответил не тем, что ждали ({error}) — размещать вслепую нельзя")


def proc_stat(pid: int) -> tuple[str, str] | None:
    """Состояние процесса и метка времени его старта из /proc. None, если процесса уже нет. По метке отличаем тот же самый процесс от нового, которому ядро выдало тот же pid."""
    with contextlib.suppress(OSError, IndexError, ValueError):
        columns = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return columns[0], columns[19]
    return None


def group_members(pgid: int) -> list[int]:
    """Живые процессы группы. Зомби не считаем, память карты они уже отдали."""
    members = []
    for proc in Path("/proc").glob("[0-9]*"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            state, _, group = (proc / "stat").read_text().rsplit(")", 1)[1].split()[:3]
            if int(group) == pgid and state != "Z":
                members.append(int(proc.name))
    return members


async def kill_orphans(settings: Settings) -> list[str]:
    """Гасит процессы vLLM, оставшиеся от прошлого запуска прокси (например, после SIGKILL), чтобы они не держали память карт. Узнаёт их по метке PROXY_INSTANCE, а процессы от старых версий прокси по VLLM_SERVED_NAME и VLLM_PORT."""
    groups: dict[int, str] = {}
    mine = os.getpgid(0)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0")
                       if b"=" in item)
            name, _, port = env.get(b"PROXY_INSTANCE", b"").decode().rpartition(":")
            if not name:
                name = env.get(b"VLLM_SERVED_NAME", b"").decode()
                port = env.get(b"VLLM_PORT", b"0").decode()
            if name in settings.models and int(port) in settings.internal_ports:
                pgid = os.getpgid(int(proc.name))
                if pgid != mine:
                    groups[pgid] = name
        except (OSError, ValueError):
            continue
    for pgid in groups:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    alive = dict(groups)
    while alive and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        for pgid in list(alive):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                alive.pop(pgid)
    for pgid in alive:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
    return [f"{name} (группа {pgid})" for pgid, name in groups.items()]


# ── процесс vLLM ───────────────────────────────────────────────────────────

class State(Enum):
    """В каком состоянии процесс vLLM. Спящая модель держит веса в RAM, а на карте остаётся только CUDA-контекст."""
    STOPPED = "остановлена"
    STARTING = "запускается"
    AWAKE = "активна"
    ASLEEP = "спит"


class Instance:
    """Одна копия модели, то есть один процесс vLLM. Умеет запускаться, засыпать, просыпаться и останавливаться."""

    def __init__(self, spec: ModelSpec, gpu_id: int, port: int, settings: Settings,
                 client: httpx.AsyncClient, index: int = 0):
        self.spec = spec
        self.gpu_id = gpu_id
        self.port = port
        self.settings = settings
        self.client = client
        self.index = index
        self.state = State.STOPPED
        self.busy = 0
        self.leaving = False
        self.can_sleep = True        # сбрасываем, если vLLM не смог уснуть, дальше такую копию просто останавливаем
        self.awake_gb: float | None = None   # сколько копия на деле занимает активной и во сне, пока нет замеров, берём конфиг
        self.tail_gb: float | None = None
        self.last_used = time.monotonic()
        self.awake_since = 0.0
        self.process: asyncio.subprocess.Process | None = None

    def __repr__(self) -> str:
        return f"<{self.label} GPU{self.gpu_id}:{self.port} {self.state.value}>"

    @property
    def label(self) -> str:
        """Имя копии для логов. У первой копии совпадает с именем модели."""
        return self.spec.name if self.index == 0 else f"{self.spec.name}#{self.index + 1}"

    @property
    def url(self) -> str:
        """Локальный адрес, на котором слушает vLLM."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        """Проверяет, что процесс запущен и ещё не завершился."""
        return self.process is not None and self.process.returncode is None

    @property
    def idle_for(self) -> float:
        """Сколько секунд модель ничего не делает. Пока есть запросы в работе, считаем ноль."""
        return 0.0 if self.busy else time.monotonic() - self.last_used

    @property
    def footprint(self) -> float:
        """Сколько памяти копия занимает активной. По замеру, а пока его нет, по vram_gb из конфига."""
        return self.awake_gb if self.awake_gb is not None else self.spec.vram_gb

    @property
    def tail(self) -> float:
        """Сколько памяти копия держит во сне. По замеру, а пока его нет, по asleep_tail_gb из конфига."""
        return self.tail_gb if self.tail_gb is not None else self.settings.asleep_tail_gb

    @property
    def holds_gb(self) -> float:
        """Сколько памяти копия занимает на карте прямо сейчас."""
        if self.state is State.AWAKE:
            return self.footprint
        if self.state is State.ASLEEP:
            return self.tail
        return 0.0

    @property
    def frees_gb(self) -> float:
        """Сколько памяти освободится, если копию выселить. Активную обычно усыпляют, и её остаток остаётся на карте."""
        if self.state is State.AWAKE and self.can_sleep:
            return max(0.0, self.footprint - self.tail)
        return self.holds_gb

    @property
    def wake_need(self) -> float:
        """Сколько памяти займёт пробуждение, то есть ровно то, что копия отдала, когда уснула."""
        return max(0.0, self.footprint - self.tail)

    @contextlib.contextmanager
    def held(self):
        """Закрывает вход в копию на время перехода и возвращает флаг как было. Просто выставить False в конце нельзя: флаг могли поставить снаружи, например когда карту забирает обучение."""
        was_leaving = self.leaving
        self.leaving = True
        try:
            yield
        finally:
            self.leaving = was_leaving

    def devices_seen(self) -> set[str]:
        """Какие CUDA_VISIBLE_DEVICES на деле видят процессы копии. Скрипт мог переписать то, что передала прокси."""
        seen = set()
        if self.process is None:
            return seen
        for pid in group_members(self.process.pid):
            with contextlib.suppress(OSError):
                for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                        seen.add(item.split(b"=", 1)[1].decode())
        return seen

    def recency(self, now: float) -> float:
        """Когда модель в последний раз работала. Занятая работает прямо сейчас."""
        return now if self.busy else self.last_used

    def resident(self, now: float) -> bool:
        """Проснулась недавно, ещё не отработала min_residency_sec, и ею продолжают пользоваться. Такую ради другой модели не выселяем, чтобы под встречной нагрузкой модели не перекидывали карту друг другу без конца. Простаивающую защищать незачем."""
        return (self.state is State.AWAKE
                and now - self.awake_since < self.settings.min_residency_sec
                and (self.busy > 0 or now - self.last_used < RESIDENCY_GRACE))

    def accepts(self, model: Any) -> bool:
        """Знает ли vLLM это имя модели сам. Алиасы передаём ему только при запуске через cmd, свой скрипт получает одно основное имя."""
        return model == self.spec.name or (self.spec.script is None and model in self.spec.aliases)

    def reserve(self) -> bool:
        """Занимает модель под запрос, если она уже готова. Внутри нет await, поэтому никто не успеет усыпить её между проверкой и занятием."""
        if self.state is State.AWAKE and self.alive and not self.leaving:
            self.busy += 1
            return True
        return False

    def release(self) -> None:
        """Отпускает модель после запроса и запоминает время, чтобы LRU знал, кого выселять первым."""
        self.busy = max(0, self.busy - 1)   # stop() обнуляет счётчик, а недоработавшие запросы всё равно придут сюда
        self.last_used = time.monotonic()

    async def start(self) -> None:
        """Запускает процесс, ждёт ответа от /health и прогревает модель. Если не поднялась, кидает ошибку, а прибирает за ней Cluster."""
        prefix = self.spec.name if self.index == 0 else f"{self.spec.name}-r{self.index + 1}"
        path = new_log(self.spec.log_dir or self.spec.cwd / "logs", prefix, self.settings.log_keep)
        self.state = State.STARTING
        log.info("запускаю %s на GPU%s, порт %s, лог %s", self.label, self.gpu_id, self.port, path)

        with open(path, "wb") as sink:
            # запускаем в отдельной группе, чтобы при остановке убить и bash-обёртку, и сам vLLM
            self.process = await asyncio.create_subprocess_exec(
                *self.command(), cwd=self.spec.cwd, env=self.environment(), start_new_session=True,
                stdout=sink, stderr=asyncio.subprocess.STDOUT)

        deadline = time.monotonic() + self.settings.start_timeout
        while time.monotonic() < deadline:
            if not self.alive:
                raise RuntimeError(f"{self.label}: процесс умер при старте, см. {path}")
            if await self.responds():
                self.state = State.AWAKE
                self.awake_since = time.monotonic()
                await self.warm_up()
                log.info("%s поднялась", self.label)
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"{self.label}: не поднялась за "
                           f"{self.settings.start_timeout:.0f} сек, см. {path}")

    def command(self) -> list[str]:
        """Собирает команду запуска. Либо свой скрипт, либо cmd из конфига с нужными флагами vLLM, включая алиасы как дополнительные имена модели."""
        if self.spec.script:
            return ["bash", str(self.spec.script)]
        return [*self.spec.cmd, "--port", str(self.port),
                "--served-model-name", self.spec.name, *self.spec.aliases, "--enable-sleep-mode"]

    def environment(self) -> dict[str, str]:
        """Собирает переменные окружения для процесса. Убирает следы окружения прокси и добавляет общие настройки, настройки модели и переменные для vLLM."""
        env = {key: value for key, value in os.environ.items()
               if key not in INHERITED_VENV}
        env["PATH"] = strip_venv_from_path(env.get("PATH", ""),
                                           os.environ.get("VIRTUAL_ENV"))
        env.update(self.settings.env)
        env.update(self.spec.env)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # нумеруем карты как NVML, иначе CUDA может назвать нулевой другую карту
        env["VLLM_SERVER_DEV_MODE"] = "1"        # без этого флага vLLM не отдаёт /sleep и /wake_up
        env["PROXY_INSTANCE"] = f"{self.spec.name}:{self.port}"   # по этой метке следующий запуск найдёт хвосты
        env["VLLM_SERVED_NAME"] = self.spec.name
        if self.spec.script:
            env["VLLM_PORT"] = str(self.port)    # скрипту порт нужен, хотя vLLM тоже читает эту переменную
        else:
            env.pop("VLLM_PORT", None)           # порт уже передан флагом, а по этой переменной vLLM заняла бы соседние порты
        env["VLLM_LOGGING_LEVEL"] = self.settings.vllm_log_level
        return env

    async def stop(self) -> None:
        """Останавливает процесс. Сначала просим завершиться по-хорошему, через 30 секунд убиваем."""
        self.leaving = True
        if self.alive:
            log.info("останавливаю %s", self.label)
            self.signal_group(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except asyncio.TimeoutError:
                log.warning("%s не завершилась за 30 сек — убиваю", self.label)
                self.signal_group(signal.SIGKILL)
                await self.process.wait()
        if self.process is not None:
            await self.wait_group(self.process.pid)
        self.process = None
        self.state = State.STOPPED
        self.busy = 0
        self.leaving = False

    async def wait_group(self, pgid: int) -> None:
        """Ждёт, пока выйдут все процессы копии. Дочерний процесс vLLM держит память карты, пока не завершится, а прокси в это время уже считает её свободной."""
        deadline = time.monotonic() + 15
        while group_members(pgid):
            if time.monotonic() > deadline:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)
                await asyncio.sleep(0.5)
                return
            await asyncio.sleep(0.2)

    def signal_group(self, sig: int) -> None:
        """Отправляет сигнал всей группе процессов, а если группы уже нет, то только самому процессу."""
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                self.process.send_signal(sig)

    async def set_sleeping(self, sleeping: bool) -> bool:
        """Усыпляет или будит модель, пока идёт переход, новые запросы в неё не пускаем. После глубокого сна заново грузит веса. Возвращает False, если переходить было не из чего, и тогда замерять нечего."""
        if self.state is not (State.AWAKE if sleeping else State.ASLEEP):
            return False
        started = time.monotonic()
        level = self.settings.sleep_level
        endpoint = f"/sleep?level={level}" if sleeping else "/wake_up"
        log.info("%s %s", "усыпляю" if sleeping else "бужу", self.label)

        with self.held():            # пока меняем состояние, новые запросы сюда не пускаем
            response = await self.client.post(f"{self.url}{endpoint}",
                                              timeout=self.settings.switch_timeout)
            if response.status_code == 404:
                raise RuntimeError(f"у vLLM нет {endpoint.split('?')[0]}, значит процесс запущен без "
                                   f"VLLM_SERVER_DEV_MODE=1, vLLM слишком старый или за портом не vLLM")
            if response.status_code != 200:
                raise RuntimeError(f"{endpoint} вернул {response.status_code}: {response.text[:300]}")
            if not sleeping and level == 2:      # на втором уровне веса выгружены полностью, поэтому грузим их заново
                await self.client.post(f"{self.url}/collective_rpc",
                                       json={"method": "reload_weights"},
                                       timeout=self.settings.switch_timeout)
                await self.client.post(f"{self.url}/reset_prefix_cache",
                                       timeout=self.settings.switch_timeout)
            await self.await_state(sleeping)
            self.state = State.ASLEEP if sleeping else State.AWAKE
            log.info("%s %s за %.1f сек", self.label,
                     "уснула" if sleeping else "проснулась", time.monotonic() - started)
            if not sleeping:
                self.awake_since = time.monotonic()
                await self.warm_up()
        return True

    async def await_state(self, sleeping: bool) -> None:
        """Опрашивает /is_sleeping, пока vLLM действительно не уснёт или не проснётся, иначе падает по таймауту. Не-JSON в ответе (чужой сервис за портом, страница ошибки) молчаливо пропускаем и ждём дальше — по таймауту это станет понятной ошибкой."""
        deadline = time.monotonic() + self.settings.switch_timeout
        while True:
            with contextlib.suppress(httpx.HTTPError, ValueError, AttributeError):
                response = await self.client.get(f"{self.url}/is_sleeping")
                if bool(response.json().get("is_sleeping", False)) is sleeping:
                    return
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.label}: состояние не сменилось за "
                                   f"{self.settings.switch_timeout:.0f} сек")
            await asyncio.sleep(0.25)

    async def responds(self) -> bool:
        """Проверяет, что vLLM отвечает на /health."""
        try:
            return (await self.client.get(f"{self.url}/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def warm_up(self) -> None:
        """Отправляет короткий запрос, чтобы vLLM собрал CUDA-графы заранее и первый настоящий запрос не тормозил."""
        try:
            response = await self.client.post(f"{self.url}/v1/completions", timeout=180,
                                              json={"model": self.spec.name, "prompt": "ok",
                                                    "max_tokens": 1})
            if response.status_code != 200:
                log.warning("прогрев %s не удался: vLLM ответил %s (%s). Если модель только для "
                            "чата или эмбеддингов, это нормально, но первый запрос будет медленным",
                            self.label, response.status_code, response.text[:200])
        except httpx.HTTPError as error:
            log.warning("прогрев %s не удался: %s", self.label, error)


# ── карта ──────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Gpu:
    """Одна видеокарта с её памятью и копиями моделей, которые на ней живут."""

    id: int
    memory: Memory
    reserve_gb: float
    instances: list[Instance] = field(default_factory=list)
    blocked: set[str] = field(default_factory=set)          # аренды обучения, которым сейчас отдана карта
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)   # на карте идёт только одно переключение за раз

    def __repr__(self) -> str:
        note = ", отдана под обучение" if self.blocked else ""
        return (f"GPU{self.id} свободно {self.memory.free_gb:.1f} "
                f"из {self.memory.total_gb:.0f} ГБ{note}")

    def foreign_gb(self) -> float:
        """Сколько памяти на карте держат не наши процессы."""
        return max(0.0, self.memory.total_gb - self.memory.free_gb - sum(i.holds_gb for i in self.instances))

    def crowded(self, newcomer: tuple[float, float], leaving: list[Instance]) -> bool:
        """Станет ли на карте тесно, если на ней поселится ещё одна модель, newcomer это её объём активной и остаток во сне. Тесно, когда хоть одна модель здесь не сможет проснуться, пока остальные спят рядом, и при переключениях кого-то придётся останавливать."""
        members = [(i.footprint, i.tail) for i in self.instances
                   if i.state is not State.STOPPED and i not in leaving] + [newcomer]
        tails = sum(tail for _, tail in members)
        room = self.memory.total_gb - self.foreign_gb() - self.reserve_gb
        return any(footprint + tails - tail > room for footprint, tail in members)

    def required(self, need: float) -> float:
        """Сколько свободной памяти нужно, чтобы модель заняла ещё need ГБ, вместе с резервом."""
        return need + self.reserve_gb

    def fits(self, need: float) -> bool:
        """Проверяет, влезет ли прямо сейчас ещё need ГБ."""
        return self.memory.free_gb >= self.required(need)

    def evictable(self, spec: ModelSpec, need: float, wait: bool = False) -> list[Instance] | None:
        """Подбирает, кого выселить, чтобы модель spec заняла ещё need ГБ. Сначала давно не используемые, приоритетные в последнюю очередь, и никого сверх необходимого. Без wait берёт только свободных и давно проснувшихся. Возвращает None, если места всё равно не хватит."""
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
        for instance in list(chosen):   # спящего соседа не гасим, если место и без него набирается
            if freed - instance.frees_gb >= short:
                chosen.remove(instance)
                freed -= instance.frees_gb
        return chosen


# ── обучение ───────────────────────────────────────────────────────────────

@dataclass
class Lease:
    """Аренда карт под обучение. Пока она есть, прокси не ставит на эти карты модели."""

    id: str
    gpus: set[int]
    owner: str
    pid: int | None
    started: str | None = None       # метка старта процесса, чтобы не принять за него новый процесс с тем же pid
    since: float = field(default_factory=time.time)

    def alive(self) -> bool:
        """Жив ли процесс обучения. Зомби считаем завершённым, чужой процесс с переиспользованным pid — тоже. Если PID не передали, аренда живёт до явного снятия."""
        if self.pid is None:
            return True
        stat = proc_stat(self.pid)
        if stat is None:
            return False
        state, started = stat
        if state == "Z":
            return False
        return self.started is None or started == self.started

    def describe(self) -> dict:
        """Аренда в виде словаря для HTTP-ответов."""
        return {"id": self.id, "owner": self.owner, "gpus": sorted(self.gpus), "pid": self.pid,
                "since": datetime.fromtimestamp(self.since).isoformat(timespec="seconds")}


# ── кластер ────────────────────────────────────────────────────────────────

class Cluster:
    """Все карты и все копии моделей. Здесь решаем, что куда поставить, кого подвинуть и когда уступить карты обучению."""

    def __init__(self, settings: Settings, probe: Probe = read_gpu_memory,
                 factory: type[Instance] = Instance,
                 control: httpx.AsyncClient | None = None):
        self.settings = settings
        self.probe = probe
        self.factory = factory
        # в тестах клиент передают снаружи, иначе создаём свой и сами закрываем
        self.control = control or httpx.AsyncClient(timeout=30)
        self._owns_control = control is None
        self.gpus: dict[int, Gpu] = {}
        self.instances: dict[str, list[Instance]] = {}
        self.model_locks = {name: asyncio.Lock() for name in settings.models}   # одну модель поднимает один запрос, остальные ждут его
        self.probe_lock = asyncio.Lock()
        self.probed_at = 0.0
        self.leases: dict[str, Lease] = {}
        self.footprints: dict[str, float] = {}     # последние замеры по моделям, чтобы новая копия сразу считалась верно
        self.tails: dict[str, float] = {}
        self.crowd_warned: set[str] = set()
        self.scaling: set[str] = set()
        self.tasks: set[asyncio.Task] = set()      # держим ссылки на фоновые задачи, чтобы их не собрал GC

    # ── публичный интерфейс ────────────────────────────────────────────────

    async def acquire(self, name: str) -> Instance:
        """Возвращает готовую копию модели, уже занятую под запрос. Если свободной копии нет, будит или запускает. Отпускать через instance.release(). Общее ожидание ограничено wait_budget, включая время в очереди за чужим холодным стартом."""
        instance = self.fastest(name)
        if instance is None:
            deadline = time.monotonic() + self.settings.wait_budget
            lock = self.model_locks[name]
            try:
                await asyncio.wait_for(lock.acquire(), self.settings.wait_budget)
            except asyncio.TimeoutError:
                raise NoRoom(f"{name}: не дождались очереди за {self.settings.wait_budget:.0f} сек, "
                             f"всё это время модель поднимал другой запрос")
            try:
                instance = self.fastest(name) or await self.ensure_ready(name, deadline)
            finally:
                lock.release()
        self.maybe_scale(instance)
        return instance

    def fastest(self, name: str) -> Instance | None:
        """Занимает наименее загруженную из готовых копий модели, если такая есть."""
        for instance in sorted(self.instances.get(name, []), key=lambda i: i.busy):
            if instance.reserve():
                return instance
        return None

    def all_instances(self) -> list[Instance]:
        """Все копии всех моделей одним списком."""
        return [instance for group in self.instances.values() for instance in group]

    def need_for(self, spec: ModelSpec, sleeper: Instance | None) -> float:
        """Сколько памяти займёт модель. Спящей копии нужно ровно то, что она отдала при засыпании. Новой копии vLLM при старте требует свою долю карты целиком, поэтому не меньше vram_gb, а если модель уже замеряли и она берёт больше, то по замеру."""
        if sleeper is not None:
            return sleeper.wake_need
        return max(spec.vram_gb, self.footprints.get(spec.name, 0.0))

    def crowds(self, gpu: Gpu, spec: ModelSpec, victims: list[Instance]) -> bool:
        """Станет ли тесно на карте, если запустить там новую копию модели. Спящих соседей, которых при этом остановят, уже не считаем."""
        gone = [v for v in victims if v.state is State.ASLEEP or not v.can_sleep]
        tail = self.tails.get(spec.name, self.settings.asleep_tail_gb)
        return gpu.crowded((self.need_for(spec, None), tail), gone)

    async def report(self) -> dict:
        """Собирает состояние карт, аренд обучения и копий моделей для /health и стартового лога."""
        try:
            await self.refresh()
        except NoRoom as error:
            return {"error": str(error)}
        return {"gpus": [repr(gpu) for gpu in self.gpus.values()],
                "training": [lease.describe() for lease in self.leases.values()],
                "models": {name: [{"state": i.state.value, "gpu": i.gpu_id, "port": i.port,
                                   "busy": i.busy, "idle_sec": round(i.idle_for),
                                   "memory_gb": round(i.holds_gb, 2),
                                   "measured": i.awake_gb is not None}
                                  for i in self.instances.get(name, [])]
                           for name in self.settings.models}}

    async def shutdown(self) -> None:
        """Гасит все модели разом и закрывает HTTP-клиент, если он наш."""
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*(i.stop() for i in self.all_instances()), return_exceptions=True)
        if self._owns_control:
            await self.control.aclose()

    # ── размещение ─────────────────────────────────────────────────────────

    async def refresh(self, max_age: float = 1.0) -> None:
        """Обновляет данные о памяти карт, если последний опрос был давно."""
        if time.monotonic() - self.probed_at < max_age:
            return
        async with self.probe_lock:
            if time.monotonic() - self.probed_at < max_age:
                return
            for index, memory in (await self.probe()).items():
                gpu = self.gpus.get(index)
                if gpu is None:
                    self.gpus[index] = Gpu(index, memory, self.settings.reserve_gb)
                else:
                    gpu.memory = memory
            self.probed_at = time.monotonic()

    def invalidate(self) -> None:
        """Сбрасывает кэш памяти, потому что мы только что сами что-то запустили или остановили."""
        self.probed_at = 0.0

    async def reap_dead(self, instance: Instance) -> Instance | None:
        """Если процесс упал сам, убирает копию из учёта и возвращает None. Для уже забытой копии тоже возвращает None."""
        if instance.alive or (instance.state is State.STARTING and instance.process is None):
            return instance          # жива или процесс как раз создаётся
        if instance in self.instances.get(instance.spec.name, []):
            log.warning("%s: процесс умер сам — вычёркиваю", instance.label)
            await self.forget(instance)
        return None

    async def ensure_ready(self, name: str, deadline: float) -> Instance:
        """Поднимает копию модели и сразу занимает её под запрос. Спящую будит на её карте, иначе запускает новую там, где выселять дешевле всего. Вызывать под model_locks[name]."""
        spec = self.settings.models[name]
        while True:
            if time.monotonic() > deadline:
                raise NoRoom(f"{name}: не дождались места за {self.settings.wait_budget:.0f} сек")
            await self.refresh()
            for instance in list(self.instances.get(name, [])):
                await self.reap_dead(instance)
            if instance := self.fastest(name):
                return instance
            if await self.wait_moving(spec, deadline) or await self.drop_stale(spec):
                continue
            gpu, sleeper = self.plan(spec)
            if gpu.lock.locked():
                await wait_unlocked(gpu.lock, deadline, f"закончится переключение на GPU{gpu.id}")
                continue             # пока ждали, расклад мог поменяться, решаем заново
            async with gpu.lock:
                try:
                    again = self.plan(spec, mine=gpu)
                except NoRoom:       # пока брали лок, карту забрали под обучение или заняли — решаем заново
                    await asyncio.sleep(0.2)
                    continue
                if again != (gpu, sleeper):
                    continue
                log.info("%s -> GPU%s: %s", name, gpu.id,
                         "бужу спящую копию" if sleeper else "запускаю копию")
                await self.free_up(gpu, spec, self.need_for(spec, sleeper), deadline)
                if sleeper is None:
                    instance = await self.launch(gpu, spec)
                elif await self.wake(sleeper):
                    instance = sleeper
                else:
                    continue         # не проснулась и уже остановлена, на следующем круге запустим заново
                if instance.reserve():
                    return instance

    async def wait_moving(self, spec: ModelSpec, deadline: float) -> bool:
        """Если копия модели сейчас засыпает, запускается или её выселяют, ждёт, пока переключение закончится. Дождаться и разбудить быстрее, чем запускать новую копию с нуля. Копии на картах, отданных под обучение, не ждём. Возвращает True, если ждал."""
        moving = [i for i in self.instances.get(spec.name, [])
                  if (i.state is State.STARTING or i.leaving) and not self.gpus[i.gpu_id].blocked]
        if not moving:
            return False
        gpu = self.gpus[moving[0].gpu_id]
        if gpu.lock.locked():
            await wait_unlocked(gpu.lock, deadline, f"{moving[0].label} закончит переключение")
        else:
            await asyncio.sleep(0.2)
        return True

    def is_stale(self, instance: Instance) -> bool:
        """Спящая копия на карте, которая не вместит модель даже пустой, потому что память держит чужой процесс."""
        return (instance.state is State.ASLEEP and self.gpus[instance.gpu_id].evictable(
            instance.spec, instance.wake_need, wait=True) is None)

    async def drop_stale(self, spec: ModelSpec) -> bool:
        """Забывает спящие копии, которым уже не проснуться на своей карте. Карты, где сейчас идёт переключение, не трогает. Возвращает True, если кого-то забыл."""
        stale = [i for i in self.instances.get(spec.name, [])
                 if self.is_stale(i) and not self.gpus[i.gpu_id].lock.locked()]
        for instance in stale:
            gpu = self.gpus[instance.gpu_id]
            log.warning("%s: на GPU%s свободно %.1f ГБ, чтобы проснуться, ей нужно %.1f, и подвинуть некого. "
                        "Память держит посторонний процесс (см. nvidia-smi), запускаю её заново там, где есть место",
                        instance.label, gpu.id, gpu.memory.free_gb, gpu.required(instance.wake_need))
            async with gpu.lock:
                await self.forget(instance)
        return bool(stale)

    def plan(self, spec: ModelSpec, mine: Gpu | None = None) -> tuple[Gpu, Instance | None]:
        """Решает, где поднять модель. Спящую копию выгоднее разбудить, чем запускать новую. Новую ставим туда, где не станет тесно, где сейчас не идёт чужой запуск, где не надо выселять приоритетных, потом где не надо ждать, и выселяем меньше и тех, кто дольше простаивал. Возвращает карту и спящую копию или None, если нужна новая."""
        replicas = self.instances.get(spec.name, [])
        sleepers = {i.gpu_id: i for i in replicas if i.state is State.ASLEEP}
        taken = {i.gpu_id for i in replicas if i.state is not State.ASLEEP}
        can_add = sum(not self.gpus[i.gpu_id].blocked and not self.is_stale(i)
                      for i in replicas) < spec.replicas
        now = time.monotonic()
        options = []
        for gpu in self.gpus.values():
            sleeper = sleepers.get(gpu.id)
            if gpu.blocked or gpu.id in taken or (sleeper is None and not can_add):
                continue
            need = self.need_for(spec, sleeper)
            victims = gpu.evictable(spec, need)
            must_wait = victims is None
            if victims is None and (victims := gpu.evictable(spec, need, wait=True)) is None:
                continue
            crowded = sleeper is None and self.crowds(gpu, spec, victims)
            busy_gpu = gpu.lock.locked() and gpu is not mine
            key = (sleeper is None, crowded, busy_gpu, sum(v.spec.priority for v in victims), must_wait,
                   len(victims), max((v.recency(now) for v in victims), default=0.0), -gpu.memory.free_gb)
            options.append((key, gpu.id, gpu, sleeper))
        if not options:
            if self.gpus and all(gpu.blocked for gpu in self.gpus.values()):
                raise TrainingInProgress(f"{spec.name}: все карты отданы под обучение, "
                                         f"модели вернутся, когда оно закончится")
            raise NoRoom(f"{spec.name}: {spec.vram_gb:.1f} ГБ не найдётся нигде. "
                         + "; ".join(map(repr, self.gpus.values())))
        key, _, gpu, sleeper = min(options, key=lambda option: option[:2])
        if key[1] and spec.name not in self.crowd_warned:
            self.crowd_warned.add(spec.name)
            log.warning("%s: ни на одной карте нет места держать её спящей рядом с остальными. "
                        "При переключениях кого-то из соседей придётся останавливать, "
                        "и его следующий запрос будет ждать холодного старта", spec.name)
        return gpu, sleeper

    async def launch(self, gpu: Gpu, spec: ModelSpec) -> Instance:
        """Запускает новую копию модели на карте, где уже освобождено место. Вызывать под gpu.lock."""
        used = {i.index for i in self.instances.get(spec.name, [])}
        index = next(n for n in itertools.count() if n not in used)
        instance = self.factory(spec, gpu.id, self.next_port(), self.settings, self.control,
                                index=index)
        instance.tail_gb = self.tails.get(spec.name)
        self.instances.setdefault(spec.name, []).append(instance)
        gpu.instances.append(instance)
        before = await self.free_now(gpu)
        try:
            await instance.start()
        except Exception:
            await self.forget(instance)
            raise
        finally:
            self.invalidate()
        if other := instance.devices_seen() - {str(gpu.id)}:
            log.error("%s: прокси поставила её на GPU%s, а процесс видит CUDA_VISIBLE_DEVICES=%s. "
                      "Уберите жёсткую карту из скрипта, иначе прокси будет путать, где чья память",
                      instance.label, gpu.id, ", ".join(sorted(other)))
        self.note_awake(instance, gpu, 0.0, before - await self.free_now(gpu))
        return instance

    async def free_now(self, gpu: Gpu) -> float:
        """Свежий замер свободной памяти на карте."""
        await self.refresh(max_age=0)
        return gpu.memory.free_gb

    def note_awake(self, instance: Instance, gpu: Gpu, base: float, grown: float) -> None:
        """Запоминает, сколько копия на деле занимает активной. base это то, что она уже держала до пробуждения. Замер, который расходится с конфигом в разы, не берём, скорее всего в это время на карте менялось что-то чужое."""
        spec = instance.spec
        total = base + grown
        if not 0.5 * spec.vram_gb <= total <= 2 * spec.vram_gb + 2:
            log.warning("%s: на GPU%s память выросла на %.1f ГБ, а ждали около %.1f. Замер не беру. "
                        "Возможно, на карте одновременно менялось что-то чужое или скрипт сам выбирает карту",
                        instance.label, gpu.id, grown, spec.vram_gb - base)
            return
        instance.awake_gb = total
        if spec.name not in self.footprints and abs(total - spec.vram_gb) > 0.3:
            log.warning("%s на деле занимает %.1f ГБ, а в конфиге vram_gb %.1f. Дальше считаю по замеру",
                        instance.label, total, spec.vram_gb)
        self.footprints[spec.name] = total

    async def free_up(self, gpu: Gpu, spec: ModelSpec, need: float, deadline: float,
                      wait: bool = True) -> None:
        """Освобождает на карте need ГБ под модель. Если после выселения всё равно мало, значит память занял чужой процесс. Вызывать под gpu.lock."""
        await self.refresh()
        if gpu.fits(need):
            return
        victims = await self.claim_victims(gpu, spec, need, deadline, wait)
        try:
            for instance in victims:
                await self.evict(instance)
        finally:
            for instance in victims:   # если выселение упало на середине, остальные не должны остаться закрытыми
                instance.leaving = False
        await self.refresh(max_age=0)
        settle = time.monotonic() + 10
        while not gpu.fits(need) and time.monotonic() < settle:   # драйвер отдаёт память остановленного процесса не мгновенно
            await asyncio.sleep(0.5)
            await self.refresh(max_age=0)
        if not gpu.fits(need):
            raise NoRoom(f"на GPU{gpu.id} после вытеснения {gpu.memory.free_gb:.1f} "
                         f"из нужных {gpu.required(need):.1f} ГБ — "
                         f"проверьте посторонние процессы")

    async def claim_victims(self, gpu: Gpu, spec: ModelSpec, need: float, deadline: float,
                            wait: bool) -> list[Instance]:
        """Выбирает, кого выселить, и закрывает им вход, чтобы очередь к ним не росла. Занятым даёт доработать, только что проснувшихся не трогает до конца min_residency_sec. Ждёт не дольше deadline, а без wait не ждёт вовсе."""
        claimed: set[Instance] = set()
        announced = False
        try:
            while True:
                now = time.monotonic()
                ready = gpu.evictable(spec, need)
                if ready is None and not wait:
                    raise NoRoom(f"на GPU{gpu.id} сейчас некого выселить без ожидания")
                victims = ready if ready is not None else gpu.evictable(spec, need, wait=True)
                if victims is None:
                    raise NoRoom(f"на GPU{gpu.id} нужно {gpu.required(need):.1f} ГБ, "
                                 f"свободно {gpu.memory.free_gb:.1f}, и подвинуть некого — "
                                 f"проверьте посторонние процессы (nvidia-smi)")
                for instance in claimed - set(victims):
                    instance.leaving = False
                claimed = {i for i in victims if not i.resident(now)}
                for instance in claimed:   # помечаем сразу без await, новый запрос сюда уже не попадёт
                    instance.leaving = True
                if ready is not None:
                    if announced:
                        log.info("%s: место на GPU%s освободилось", spec.name, gpu.id)
                    return victims
                waiting = ", ".join(i.label for i in victims if i.busy or i.resident(now))
                if now > deadline:
                    raise NoRoom(f"{spec.name}: не дождались места на GPU{gpu.id}, "
                                 f"карту занимали: {waiting}")
                if not announced:
                    log.info("%s ждёт места на GPU%s: %s дорабатывает запросы или только что проснулась",
                             spec.name, gpu.id, waiting)
                    announced = True
                await asyncio.sleep(0.5)
                await self.refresh(max_age=0)
        except BaseException:
            for instance in claimed:
                instance.leaving = False
            raise

    async def evict(self, instance: Instance) -> None:
        """Забирает память у копии. Единственную активную копию модели усыпляет, а спящую, лишнюю или не умеющую спать останавливает и убирает из учёта."""
        alone = len(self.instances.get(instance.spec.name, [])) == 1
        if instance.state is State.AWAKE and alone and instance.can_sleep:
            if await self.put_to_sleep(instance):
                return
        await self.forget(instance)

    async def put_to_sleep(self, instance: Instance) -> bool:
        """Усыпляет копию и замеряет, сколько она отдала и сколько осталось. Если vLLM уснуть не смог, запоминает это и возвращает False, тогда при нехватке места такую копию просто останавливаем."""
        gpu = self.gpus[instance.gpu_id]
        with instance.held():     # вход закрываем до первого await, иначе запрос успеет занять копию, пока мы опрашиваем карту
            if instance.busy:     # по пути вытеснения такого не бывает, а уборка могла прийти в ровно занятую копию
                return True
            try:
                before = await self.free_now(gpu)
                if not await instance.set_sleeping(True):
                    return True   # переходить было не из чего, мерить нечего
                freed = await self.free_now(gpu) - before
                if instance.awake_gb is None:           # активную не замеряли, зато точно знаем, сколько она отдала
                    if freed > 0.3 * instance.spec.vram_gb:
                        instance.awake_gb = freed + instance.tail
                        self.footprints[instance.spec.name] = instance.awake_gb
                elif 0 <= instance.awake_gb - freed <= 0.5 * instance.awake_gb:
                    measured = instance.awake_gb - freed
                    if (instance.spec.name not in self.tails
                            and abs(measured - self.settings.asleep_tail_gb) > 0.3):
                        log.warning("%s во сне держит %.1f ГБ, а asleep_tail_gb в конфиге %.1f. "
                                    "Дальше считаю по замеру, но размещение при старте прокси "
                                    "решается по конфигу — поправьте его",
                                    instance.label, measured, self.settings.asleep_tail_gb)
                    instance.tail_gb = self.tails[instance.spec.name] = measured
                return True
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                instance.can_sleep = False
                log.warning("%s не засыпает (%s). Спать её больше не укладываю, при нехватке места "
                            "буду останавливать, и следующий запуск будет холодным", instance.label, error)
                return False
            finally:
                self.invalidate()

    async def wake(self, instance: Instance) -> bool:
        """Будит спящую копию и замеряет, сколько она заняла. Если не вышло, останавливает её, чтобы следующая попытка подняла модель с нуля. Возвращает True, если проснулась."""
        gpu = self.gpus[instance.gpu_id]
        try:
            before = await self.free_now(gpu)
            tail = instance.tail          # запоминаем до перехода: это база, поверх которой она доберёт память
            if not await instance.set_sleeping(False):
                return instance.state is State.AWAKE
            self.note_awake(instance, gpu, tail, before - await self.free_now(gpu))
            return True
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            log.warning("%s не проснулась (%s), останавливаю и подниму заново", instance.label, error)
            await self.forget(instance)
            return False
        finally:
            self.invalidate()

    async def forget(self, instance: Instance) -> None:
        """Останавливает копию и полностью убирает её из учёта. Следующий запрос поднимет модель заново."""
        await instance.stop()
        group = self.instances.get(instance.spec.name, [])
        if instance in group:
            group.remove(instance)
        if not group:
            self.instances.pop(instance.spec.name, None)
        gpu = self.gpus.get(instance.gpu_id)
        if gpu and instance in gpu.instances:
            gpu.instances.remove(instance)
        self.invalidate()

    def next_port(self) -> int:
        """Ищет свободный внутренний порт для новой копии. Сначала с шагом PORT_STEP, чтобы не попасть на порты, которые соседний vLLM занимает сразу за своим, и только потом любые."""
        mine = {instance.port for instance in self.all_instances()}
        ports = self.settings.internal_ports
        for port in [*ports[::PORT_STEP], *ports]:
            if port not in mine and port_is_free(port):
                return port
        raise NoRoom("свободных внутренних портов нет: расширьте internal_ports "
                     "или проверьте процессы с прошлых запусков (ss -tlnp)")

    # ── масштабирование ────────────────────────────────────────────────────

    def maybe_scale(self, instance: Instance) -> None:
        """Если даже наименее загруженная копия занята сильнее scale_up_busy, в фоне добавляет модели ещё одну копию, пока не упрётся в replicas."""
        spec = instance.spec
        working = [i for i in self.instances.get(spec.name, [])
                   if i.state in (State.AWAKE, State.STARTING)]
        if (instance.busy < self.settings.scale_up_busy or len(working) >= spec.replicas
                or spec.name in self.scaling):
            return
        self.scaling.add(spec.name)
        task = asyncio.create_task(self.scale_up(spec))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def scale_target(self, spec: ModelSpec, mine: Gpu | None = None) -> tuple[Gpu, Instance | None] | None:
        """Карта для ещё одной копии. Годится только та, где не надо ждать и не нужно выселять приоритетных. Спящую копию лучше разбудить, чем запускать новую."""
        replicas = self.instances.get(spec.name, [])
        sleepers = {i.gpu_id: i for i in replicas if i.state is State.ASLEEP}
        taken = {i.gpu_id for i in replicas if i.state is not State.ASLEEP}
        now = time.monotonic()
        options = []
        for gpu in self.gpus.values():
            sleeper = sleepers.get(gpu.id)
            if (gpu.blocked or gpu.id in taken or (gpu.lock.locked() and gpu is not mine)
                    or (sleeper is None and len(replicas) >= spec.replicas)):
                continue
            victims = gpu.evictable(spec, self.need_for(spec, sleeper))
            if (victims is None or any(v.spec.priority for v in victims)
                    or (sleeper is None and self.crowds(gpu, spec, victims))):
                continue
            key = (sleeper is None, len(victims),
                   max((v.recency(now) for v in victims), default=0.0), -gpu.memory.free_gb)
            options.append((key, gpu.id, gpu, sleeper))
        if not options:
            return None
        _, _, gpu, sleeper = min(options, key=lambda option: option[:2])
        return gpu, sleeper

    async def scale_up(self, spec: ModelSpec) -> None:
        """Добавляет модели ещё одну рабочую копию, будит спящую или запускает новую. Если подходящей карты нет, ничего не делает."""
        try:
            await self.refresh()
            target = self.scale_target(spec)
            if target is None:
                log.info("%s: нагрузка высокая, но места для ещё одной копии сейчас нет", spec.name)
                return
            gpu, sleeper = target
            async with gpu.lock:
                await self.refresh()
                if self.scale_target(spec, mine=gpu) != target:
                    return
                log.info("%s: нагрузка высокая, %s на GPU%s", spec.name,
                         "бужу спящую копию" if sleeper else "добавляю копию", gpu.id)
                await self.free_up(gpu, spec, self.need_for(spec, sleeper), time.monotonic(), wait=False)
                if sleeper is not None:
                    await self.wake(sleeper)
                else:
                    await self.launch(gpu, spec)
        except Exception as error:
            log.warning("%s: не удалось добавить копию: %s", spec.name, error)
        finally:
            self.scaling.discard(spec.name)

    # ── обучение ───────────────────────────────────────────────────────────

    def usable_gb(self, gpu: Gpu) -> float:
        """Сколько памяти карты достанется обучению, когда прокси уберёт с неё свои модели. Чужие процессы никуда не денутся, резерв оставляем на фрагментацию."""
        return max(0.0, gpu.memory.total_gb - gpu.foreign_gb() - self.settings.reserve_gb)

    def pick_gpus(self, vram_gb: float) -> list[int]:
        """Набирает карты под обучение, которому нужно vram_gb ГБ. Берёт те, что меньше всего мешают моделям: сначала без приоритетных, потом где меньше активных копий, потом те, что просторнее. Карты одинакового объёма так же дают и наименьшее их число."""
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

    async def start_training(self, gpus: list[int] | None, owner: str, pid: int | None,
                             vram_gb: float | None = None) -> Lease:
        """Отдаёт карты под обучение. Карты можно назвать списком, попросить по объёму через vram_gb или не указывать ничего и забрать все. Сразу закрывает на них вход моделям, даёт текущим запросам доработать не дольше drain_timeout, гасит модели и возвращает аренду."""
        await self.refresh()
        if vram_gb is not None:
            if gpus is not None:
                raise ValueError("укажите либо gpus, либо vram_gb, но не оба сразу")
            if vram_gb <= 0:
                raise ValueError("vram_gb должно быть больше нуля")
            targets = set(self.pick_gpus(vram_gb))
        elif gpus is None:
            targets = set(self.gpus)          # ничего не просили — забираем всё
        elif not gpus:
            raise ValueError("gpus пуст: перечислите карты, попросите объём через vram_gb "
                             "или уберите поле совсем, чтобы забрать все карты")
        else:
            targets = set(gpus)
        if unknown := targets - set(self.gpus):
            raise ValueError(f"карт {sorted(unknown)} нет, есть {sorted(self.gpus)}")
        stat = proc_stat(pid) if pid is not None else None
        if pid is not None and stat is None:
            raise ValueError(f"процесса {pid} нет — проверьте pid, иначе аренду снимет первая же уборка")
        lease = Lease(uuid.uuid4().hex[:8], targets, owner, pid, stat[1] if stat else None)
        self.leases[lease.id] = lease
        for gpu_id in targets:
            gpu = self.gpus[gpu_id]
            gpu.blocked.add(lease.id)
            for instance in gpu.instances:   # новые запросы сюда больше не пойдут
                instance.leaving = True
        log.warning("обучение %s (%s) забирает GPU %s", lease.id, owner, sorted(targets))
        try:
            # return_exceptions, иначе первая же ошибка вернула бы управление, пока остальные карты ещё дренируются
            done = await asyncio.gather(*(self.vacate(self.gpus[gpu_id]) for gpu_id in sorted(targets)),
                                        return_exceptions=True)
        except BaseException:
            await self.end_training(lease.id)
            raise
        if failed := [item for item in done if isinstance(item, BaseException)]:
            await self.end_training(lease.id)
            raise failed[0]
        return lease

    async def vacate(self, gpu: Gpu) -> None:
        """Гасит все модели на карте. Текущим запросам даёт доработать, но не дольше drain_timeout."""
        async with gpu.lock:
            for instance in gpu.instances:
                instance.leaving = True
            deadline = time.monotonic() + self.settings.drain_timeout
            while any(i.busy for i in gpu.instances) and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
            if busy := [i.label for i in gpu.instances if i.busy]:
                log.warning("GPU%s: %s не доработали за %.0f сек, гашу вместе с запросами",
                            gpu.id, ", ".join(busy), self.settings.drain_timeout)
            for instance in list(gpu.instances):
                await self.forget(instance)
            await self.refresh(max_age=0)

    async def end_training(self, lease_id: str) -> Lease:
        """Снимает аренду и возвращает карты моделям. Приоритетные поднимутся сами при ближайшей уборке, остальные по первому запросу."""
        lease = self.leases.pop(lease_id)
        for gpu_id in lease.gpus:
            if gpu := self.gpus.get(gpu_id):
                gpu.blocked.discard(lease_id)
                if not gpu.blocked and not gpu.lock.locked():
                    for instance in gpu.instances:
                        instance.leaving = False
        log.warning("обучение %s (%s) закончилось, GPU %s снова доступны моделям",
                    lease.id, lease.owner, sorted(lease.gpus))
        self.invalidate()
        return lease

    async def check_leases(self) -> None:
        """Снимает аренды, чей процесс обучения уже завершился, например упало ядро ноутбука."""
        for lease in list(self.leases.values()):
            if not lease.alive():
                log.warning("процесс обучения %s (pid %s) завершился, а аренду не сняли, снимаю сам",
                            lease.id, lease.pid)
                await self.end_training(lease.id)

    # ── фоновое обслуживание ───────────────────────────────────────────────

    async def preload(self) -> None:
        """Поднимает все модели по очереди при старте прокси. Приоритетные идут первыми и занимают свои карты, остальные делят те, где приоритетных нет. В конце приоритетные будим ещё раз, если на одной карте их всё же пришлось усыпить."""
        specs = sorted(self.settings.models.values(), key=lambda s: not s.priority)
        for spec in specs + [s for s in specs if s.priority]:
            try:
                (await self.acquire(spec.name)).release()
                log.info("%s готова", spec.name)
            except Exception as error:
                log.error("%s не поднялась: %s", spec.name, error)

    async def housekeeping(self) -> None:
        """Раз в 15 секунд наводит порядок."""
        while True:
            await asyncio.sleep(15)
            await self.tidy()

    async def tidy(self) -> None:
        """Снимает брошенные аренды, гасит лишнее и поднимает приоритетные. Ошибка одного шага не мешает остальным."""
        for step in (self.check_leases, self.reap, self.restore_priority):
            try:
                await step()
            except Exception as error:
                log.warning("уборка (%s) не удалась: %s", step.__name__, error)

    async def reap(self) -> None:
        """Лишние простаивающие копии останавливает. Остальные долго простаивающие усыпляет, а долго спящие останавливает совсем. Приоритетные держит, пока у них одна копия. Карты, где сейчас идёт переключение, не трогает."""
        for instance in sorted(self.all_instances(),
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
                log.warning("%s: уборка не удалась: %s", instance.label, error)

    async def reap_one(self, instance: Instance, gpu: Gpu) -> None:
        """Решает судьбу одной простаивающей копии. Вызывать под gpu.lock."""
        idle = instance.idle_for
        extra = len(self.instances.get(instance.spec.name, [])) > 1
        if instance.busy or instance.leaving or instance not in gpu.instances:
            return
        if extra and idle > self.settings.idle_sleep_sec:
            log.info("%s: лишняя копия простаивает %.0f сек, останавливаю", instance.label, idle)
            await self.forget(instance)
        elif instance.spec.priority:
            return
        elif (instance.state is State.AWAKE and instance.can_sleep
              and idle > self.settings.idle_sleep_sec):
            log.info("%s простаивает %.0f сек — усыпляю", instance.label, idle)
            await self.put_to_sleep(instance)
        elif idle > self.settings.idle_stop_sec and (instance.state is State.ASLEEP
                                                     or not instance.can_sleep):
            log.info("%s не нужна уже %.0f сек — останавливаю", instance.label, idle)
            await self.forget(instance)

    async def restore_priority(self) -> None:
        """Держит приоритетные модели наготове. Спящую будит, а остановленную (например, после обучения) запускает, если на карте есть место без выселения и никто не ждёт ответа."""
        if any(instance.busy for instance in self.all_instances()):
            return
        for spec in self.settings.models.values():
            replicas = self.instances.get(spec.name, [])
            if not spec.priority or any(i.state in (State.AWAKE, State.STARTING) for i in replicas):
                continue
            await self.refresh()
            sleeper = next((i for i in replicas if i.state is State.ASLEEP), None)
            if sleeper is not None:
                gpu = self.gpus[sleeper.gpu_id]
                if gpu.blocked or gpu.lock.locked() or not gpu.fits(sleeper.wake_need):
                    continue
                async with gpu.lock:
                    log.info("на GPU%s есть место — бужу приоритетную %s", gpu.id, spec.name)
                    await self.wake(sleeper)
            elif not replicas:
                free = [g for g in self.gpus.values()
                        if not g.blocked and not g.lock.locked() and g.fits(self.need_for(spec, None))]
                if not free:
                    continue
                gpu = max(free, key=lambda g: g.memory.free_gb)
                async with gpu.lock:
                    log.info("на GPU%s есть место — запускаю приоритетную %s", gpu.id, spec.name)
                    await self.launch(gpu, spec)


# ── HTTP ───────────────────────────────────────────────────────────────────

class QuietServer(uvicorn.Server):
    """uvicorn без своей обработки сигналов. Сигналы ловит main и гасит всё по порядку, иначе после SIGTERM процессы vLLM остаются висеть на картах."""

    def install_signal_handlers(self) -> None:      # uvicorn старше 0.29
        pass

    @contextlib.contextmanager
    def capture_signals(self):                      # uvicorn 0.29 и новее
        yield


class Proxy:
    """Принимает запросы на внешних портах и пересылает их нужной модели. Служебные ручки живут на отдельном порту, который слушает только 127.0.0.1."""

    def __init__(self, settings: Settings, cluster: Cluster):
        self.settings = settings
        self.cluster = cluster
        self.counter = itertools.count(1)
        # лимит соединений снят, чтобы vLLM мог собрать большой батч, таймаут чтения задаётся в конфиге
        self.inference = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.read_timeout, connect=10),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))
        self.pumps: set[asyncio.Task] = set()      # держим ссылки на задачи стримов, чтобы их не собрал GC

    def application(self, port: int) -> FastAPI:
        """Создаёт FastAPI-приложение для одного внешнего порта."""
        app = FastAPI()
        created = int(time.time())

        @app.get("/v1/models")
        async def models(request: Request):
            """Список моделей этого порта в формате OpenAI. Отвечаем сами из конфига, чтобы не будить модели ради списка."""
            self.check_key(request)
            names = [alias for name in self.settings.on_port(port)
                     for alias in (name, *self.settings.models[name].aliases)]
            return {"object": "list", "data": [{"id": name, "object": "model", "created": created,
                                                "owned_by": "vllm"} for name in names]}

        @app.get("/v1/{path:path}")
        @app.post("/v1/{path:path}")
        async def handle(path: str, request: Request):
            """Принимает запросы к OpenAI-совместимому API. Пути с точками и процентами не пускаем, иначе через них можно добраться до служебных ручек vLLM."""
            self.check_key(request)
            if not SAFE_PATH.fullmatch(path):
                raise HTTPException(404, "такого эндпоинта нет")
            return await self.forward(port, path, request)

        return app

    def admin_application(self) -> FastAPI:
        """Служебное приложение для 127.0.0.1. Состояние карт и аренда карт под обучение."""
        app = FastAPI()

        @app.get("/health")
        async def health():
            """Отдаёт состояние карт, аренд и моделей."""
            return await self.cluster.report()

        @app.get("/admin/training")
        async def leases():
            """Показывает текущие аренды под обучение."""
            return {"leases": [lease.describe() for lease in self.cluster.leases.values()]}

        @app.post("/admin/training")
        async def start_training(request: Request):
            """Отдаёт карты под обучение. Отвечает, когда модели на них уже погашены."""
            raw = await request.body()
            body = await self.parse(raw) if raw.strip() else {}
            if unknown := sorted(set(body) - {"gpus", "pid", "owner", "vram_gb"}):
                raise HTTPException(400, f"непонятные поля {', '.join(unknown)}: "
                                         f"допустимы gpus, vram_gb, pid, owner")
            gpus, pid, need = body.get("gpus"), body.get("pid"), body.get("vram_gb")
            if gpus is not None and not (isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)):
                raise HTTPException(400, "gpus должен быть списком номеров карт или null")
            if pid is not None and not isinstance(pid, int):
                raise HTTPException(400, "pid должен быть числом или null")
            if need is not None and (isinstance(need, bool) or not isinstance(need, (int, float))):
                raise HTTPException(400, "vram_gb должен быть числом ГБ или null")
            try:
                lease = await self.cluster.start_training(
                    gpus, str(body.get("owner") or "обучение"), pid, float(need) if need else None)
            except ValueError as error:
                raise HTTPException(400, str(error))
            got = sum(self.cluster.gpus[g].memory.free_gb for g in lease.gpus)
            return {**lease.describe(), "free_gb": round(got, 1),
                    "memory": [repr(self.cluster.gpus[g]) for g in sorted(lease.gpus)]}

        @app.delete("/admin/training/{lease_id}")
        async def end_training(lease_id: str):
            """Возвращает карты моделям."""
            if lease_id not in self.cluster.leases:
                raise HTTPException(404, f"аренды {lease_id} нет")
            return (await self.cluster.end_training(lease_id)).describe()

        return app

    def check_key(self, request: Request) -> None:
        """Если в конфиге задан api_key, пускает только запросы с заголовком Authorization и этим ключом."""
        key = self.settings.api_key
        if key and not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                                           f"Bearer {key}".encode()):
            raise HTTPException(401, "нужен заголовок Authorization: Bearer <api_key>")

    async def forward(self, port: int, path: str, request: Request) -> Response:
        """Обрабатывает запрос целиком. Определяет модель, ждёт для неё место и возвращает ответ vLLM."""
        tag = f"[#{next(self.counter)}]"
        started = time.monotonic()
        query = request.url.query

        if request.method == "GET":     # у GET нет тела, поэтому модель понятна только на одномодельном порту
            name = self.settings.resolve(None, port)
            if name is None:
                raise HTTPException(400, f"на порту {port} несколько моделей "
                                         f"({', '.join(self.settings.on_port(port))}), "
                                         f"а GET не несёт поля model — обращайтесь POST-ом")
            log.info("%s :%s GET /v1/%s -> %s", tag, port, path, name)
            return await self.send(name, "GET", path, query, None, None, tag, started)

        raw = await request.body()
        body = await self.parse(raw)
        name = self.settings.resolve(body.get("model"), port)
        if name is None:
            raise HTTPException(400, f"на порту {port} доступны: "
                                     f"{', '.join(self.settings.on_port(port))}")
        log.info("%s :%s POST /v1/%s -> %s", tag, port, path, name)
        return await self.send(name, "POST", path, query, raw, body, tag, started)

    @staticmethod
    async def parse(raw: bytes) -> dict:
        """Разбирает тело запроса и проверяет, что это JSON-объект."""
        try:
            body = await off_loop(json.loads, raw, size=len(raw))
        except ValueError:
            raise HTTPException(400, "тело запроса должно быть корректным JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "ожидается JSON-объект")
        return body

    async def send(self, name: str, method: str, path: str, query: str, raw: bytes | None,
                   body: dict | None, tag: str, started: float) -> Response:
        """Занимает модель и пересылает ей запрос. Обычный ответ дочитывает целиком, стрим отдаёт по мере генерации. Ошибки превращает в понятные HTTP-коды."""
        instance = upstream = None
        try:
            instance = await self.cluster.acquire(name)
            if (waited := time.monotonic() - started) > 1:
                log.info("%s место получено за %.1f сек", tag, waited)
            url = f"{instance.url}/v1/{path}" + (f"?{query}" if query else "")
            if raw is None:
                request = self.inference.build_request(method, url)
            else:
                if instance.accepts(body.get("model")):
                    content = raw      # имя модели vLLM знает сама, тело уходит без пересборки
                else:
                    content = await off_loop(encode_json, {**body, "model": name}, size=len(raw))
                request = self.inference.build_request(method, url, content=content,
                                                       headers={"content-type": "application/json"})
            upstream = await self.inference.send(request, stream=True)
            if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                response = self.relay(upstream, instance, tag, started)
                instance = upstream = None      # теперь модель и соединение отпустит relay
                return response
            await upstream.aread()
            return self.finish(upstream, tag, started)
        except NoRoom as error:
            log.error("%s %s", tag, error)
            retry = "300" if isinstance(error, TrainingInProgress) else "30"
            raise HTTPException(503, str(error), headers={"Retry-After": retry})
        except (httpx.HTTPError, RuntimeError, OSError, ValueError) as error:
            log.error("%s %s не отработала: %s", tag, name, error)
            raise HTTPException(502, f"{name}: {error}")
        finally:
            if instance is not None:
                instance.release()
            if upstream is not None:
                await upstream.aclose()

    def relay(self, upstream: httpx.Response, instance: Instance,
              tag: str, started: float) -> StreamingResponse:
        """Отдаёт стрим клиенту по мере генерации. Модель отпускаем, только когда vLLM допишет ответ, даже если клиент ушёл раньше, иначе её могут усыпить посреди генерации. Клиенту, который не успевает читать, копим не больше STREAM_BUFFER байт и дальше обрываем его, а модели даём договорить."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        buffered = 0

        async def pump() -> None:
            nonlocal buffered
            dropped = False
            try:
                async for chunk in upstream.aiter_bytes():
                    if dropped:
                        continue          # клиента уже нет, но upstream дочитываем, чтобы не рвать генерацию
                    if buffered + len(chunk) > STREAM_BUFFER:
                        dropped = True
                        log.warning("%s клиент не забирает стрим (накопилось %.1f МБ) — обрываю его, "
                                    "модели даю дописать ответ", tag, buffered / 2 ** 20)
                        continue
                    buffered += len(chunk)
                    queue.put_nowait(chunk)
                log.info("%s стрим за %.1f сек", tag, time.monotonic() - started)
            except httpx.HTTPError as error:
                log.error("%s стрим оборвался: %s", tag, error)
            finally:
                instance.release()
                queue.put_nowait(None)
                await upstream.aclose()

        task = asyncio.create_task(pump())
        self.pumps.add(task)
        task.add_done_callback(self.pumps.discard)

        async def chunks():
            nonlocal buffered
            while (chunk := await queue.get()) is not None:
                buffered -= len(chunk)
                yield chunk

        # без этих заголовков nginx копит стрим у себя и отдаёт клиенту одним куском
        return StreamingResponse(chunks(), upstream.status_code,
                                 media_type=upstream.headers["content-type"],
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    @staticmethod
    def finish(response: httpx.Response, tag: str, started: float) -> Response:
        """Отдаёт клиенту ответ vLLM без изменений и пишет в лог число токенов."""
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            log.warning("%s не-JSON ответ (%s)", tag, response.status_code)
        elif response.status_code == 200:
            try:                       # тело могло оборваться на полуслове, но клиенту отдаём что есть
                payload = response.json()
            except ValueError as error:
                log.warning("%s модель вернула битый JSON: %s", tag, error)
                payload = {}
            usage = (payload.get("usage") or {}) if isinstance(payload, dict) else {}
            log.info("%s ответ за %.1f сек, токенов: %s + %s", tag,
                     time.monotonic() - started, usage.get("prompt_tokens", "?"),
                     usage.get("completion_tokens", "?"))
        else:
            log.warning("%s модель вернула %s: %s", tag, response.status_code, response.text[:300])
        return Response(response.content, response.status_code,
                        media_type=content_type)

    async def run(self, stop: asyncio.Event) -> None:
        """Поднимает внешние порты и служебный порт на 127.0.0.1 и работает, пока не придёт stop или не упадёт один из серверов."""
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
    """Настраивает логи. В файл пишем подробно, в консоль только предупреждения. Файл режется по 50 МБ, старые логи сверх keep удаляются."""
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
    """Печатает при старте, какие порты за какими моделями, где служебный порт, что с картами и как грузятся модели."""
    lines = [f"лог: {log_path}"]
    lines += [f"порт :{port} -> {', '.join(settings.on_port(port))}"
              for port in settings.listen]
    lines.append(f"служебный порт: http://127.0.0.1:{settings.admin_port} (/health, /admin/training)")
    state = await cluster.report()
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

    text = "\n".join(lines)
    print(text, flush=True)
    log.info("%s", text)
    if error:
        log.error("карты не опрошены: %s", error)
    if not settings.api_key:
        log.warning("api_key не задан: внешние порты никого не проверяют")
    missing = [name for name, spec in settings.models.items()
               if "HF_HOME" not in {**os.environ, **settings.env, **spec.env}]
    if missing:
        log.warning("HF_HOME не задан для %s — vLLM может заново скачивать чекпоинты",
                    ", ".join(missing))


async def main() -> None:
    """Загружает конфиг, гасит хвосты прошлого запуска, запускает кластер, фоновые задачи и HTTP, а по SIGINT, SIGTERM или SIGHUP всё корректно гасит."""
    settings = Settings.load(os.getenv("CONFIG", HERE / "config.yaml"))
    path = setup_logging(Path(os.getenv("LOG_DIR", HERE / "logs")), settings.log_keep)
    busy = [port for port in settings.listen if not port_is_free(port)]
    if not port_is_free(settings.admin_port, "127.0.0.1"):
        busy.append(settings.admin_port)
    if busy:
        raise SystemExit(f"порты {busy} уже заняты, прокси уже запущена?")
    killed = await kill_orphans(settings)
    cluster = Cluster(settings)

    stop = asyncio.Event()

    def on_signal() -> None:
        if stop.is_set():          # второй сигнал, значит ждать не хотят. Хвосты уберёт следующий запуск
            log.warning("повторный сигнал, выхожу не дожидаясь остановки моделей")
            os._exit(1)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) == signal.SIG_IGN:
            continue           # сигнал велели не замечать, например запуск через nohup, не перехватываем
        loop.add_signal_handler(sig, on_signal)

    await announce(settings, cluster, path, killed)
    housekeeper = asyncio.create_task(cluster.housekeeping())
    warmer = asyncio.create_task(cluster.preload()) if settings.preload else None
    try:
        await Proxy(settings, cluster).run(stop)
    finally:
        housekeeper.cancel()
        if warmer:
            warmer.cancel()
        await cluster.shutdown()
        log.info("прокси остановлена")


def run() -> None:
    """Точка входа. Использует uvloop, если он установлен, иначе обычный asyncio."""
    try:
        import uvloop
        runner = uvloop.run
    except ImportError:
        runner = asyncio.run
    with contextlib.suppress(KeyboardInterrupt):
        runner(main())


if __name__ == "__main__":
    run()
