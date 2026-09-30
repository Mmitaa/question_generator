"""Разбор конфига, мелкие утилиты и маршрутизация по портам."""
import socket
import sys

from fixture import ROOT, SCRIPT, check, expect_exit, load, one_model, production, report
import proxy as P

print("\n[конфиг]")
s = load(one_model())
check("валидный конфиг грузится", s.models["m"].vram_gb == 10)
check("wait_budget = queue + start", s.wait_budget == s.queue_timeout + s.start_timeout)
bad = {"cmd": "v", "cwd": str(ROOT / "model"), "vram_gb": 1, "port": 8000}
expect_exit("опечатка в поле модели", one_model(models={"m": {**bad, "prioriti": True}}),
            "непонятные поля")
expect_exit("нет обязательного cwd", one_model(models={"m": {"cmd": "v", "vram_gb": 1, "port": 8000}}),
            "обязательного поля cwd")
expect_exit("опечатка в корне", one_model(idle_slep_sec=5), "непонятные поля")
expect_exit("плохой sleep_level", one_model(sleep_level=3), "sleep_level")
expect_exit("tensor-parallel ловится",
            one_model(models={"m": {**bad, "cmd": "vllm serve X --tensor-parallel-size 2"}}), "одну")
expect_exit("tp через знак равно", one_model(models={"m": {**bad, "cmd": "vllm serve X -tp=4"}}), "одну")
check("tp 1 проходит", load(one_model(models={"m": {**bad, "cmd": "vllm serve X -tp 1"}}))
      .models["m"].cmd[-1] == "1")
expect_exit("портов меньше, чем копий",
            one_model(internal_ports=[9000, 9000], models={"m": {**bad, "replicas": 3}}),
            "internal_ports вмещает")
expect_exit("битый internal_ports", one_model(internal_ports=5), "парой чисел")
expect_exit("и script, и cmd сразу",
            one_model(models={"m": {**bad, "script": str(SCRIPT)}}), "либо script, либо cmd")
expect_exit("несколько ошибок разом сообщаются вместе",
            one_model(sleep_level=7, reserve_gb=-1), "; ")

print("\n[маршрутизация по портам]")
prod = load(production())
check("порты разобраны", {p: set(prod.on_port(p)) for p in prod.listen}
      == {5000: {"test-gen", "detector-4b"}, 6000: {"grade-ai", "detector-14b"}},
      {p: prod.on_port(p) for p in prod.listen})
check("алиас ведёт к своей модели", prod.resolve("Qwen/Qwen3-14B-AWQ", 6000) == "detector-14b")
check("чужая модель на порту -> None", prod.resolve("test-gen", 6000) is None)
check("без поля model на двухмодельном порту -> None", prod.resolve(None, 6000) is None)
check("на одномодельном порту поле model не важно", s.resolve(None, 8000) == "m")

print("\n[Lease и proc_stat]")
import os
stat = P.proc_stat(os.getpid())
check("proc_stat читает свой процесс", stat is not None and stat[0] in "RSD")
check("proc_stat про несуществующий pid", P.proc_stat(4_000_000) is None)
check("аренда своего процесса жива", P.Lease("a", {0}, "я", os.getpid(), stat[1]).alive())
check("pid переиспользован — аренда мертва", not P.Lease("b", {0}, "я", os.getpid(), "9" * 9).alive())
check("без pid аренда живёт", P.Lease("c", {0}, "я", None).alive())

print("\n[held и release]")
inst = P.Instance(s.models["m"], 0, 9000, s, None)
inst.leaving = True
with inst.held():
    check("внутри held вход закрыт", inst.leaving)
check("снаружи флаг восстановлен в True", inst.leaving)
inst.leaving = False
with inst.held():
    pass
check("и в False тоже", not inst.leaving)
inst.busy = 0
inst.release()
check("release не уводит busy в минус", inst.busy == 0)

print("\n[маршруты и порты]")
app = P.Proxy(s, None).application(8000)
routes = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "methods")}
check("GET /v1/models отдельно", ("/v1/models", ("GET",)) in routes)
check("catch-all принимает GET и POST", ("/v1/{path:path}", ("GET",)) in routes
      and ("/v1/{path:path}", ("POST",)) in routes, sorted(routes))
check("SAFE_PATH режет точки", not P.SAFE_PATH.fullmatch("../sleep"))
check("SAFE_PATH пускает обычный путь", bool(P.SAFE_PATH.fullmatch("chat/completions")))
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
check("занятый порт виден занятым", not P.port_is_free(srv.getsockname()[1], "127.0.0.1"))
srv.close()

sys.exit(report())
