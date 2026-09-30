"""Сквозная проверка HTTP-слоя: реальный uvicorn-фейк вместо vLLM."""
import asyncio
import json
import sys

from fixture import ROOT, SCRIPT, check, load, one_model, report
import proxy as P
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

seen: dict = {}
fake = FastAPI()

@fake.get("/v1/{path:path}")
async def fake_get(path: str, request: Request):
    seen["get"] = (path, str(request.url.query))
    return JSONResponse({"object": "list", "path": path, "query": str(request.url.query)})

@fake.post("/v1/{path:path}")
async def fake_post(path: str, request: Request):
    body = json.loads(await request.body())
    seen["post"] = (path, str(request.url.query), body)
    if body.get("stream"):
        async def gen():
            for i in range(5):
                yield f"data: {i}\n\n".encode()
                await asyncio.sleep(0.01)
            yield b"data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    if body.get("broken"):
        return P.Response(b'{"usage": {"prompt', 200, media_type="application/json")
    return JSONResponse({"model": body["model"], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

class Wired(P.Instance):
    """Копия, которая уже «поднята» и указывает на фейковый vLLM."""
    upstream_port = 0
    async def start(self):
        self.port = Wired.upstream_port
        self.process = type("P", (), {"returncode": None, "pid": 1})()
        self.state = P.State.AWAKE
    def devices_seen(self): return set()
    async def stop(self):
        self.process = None; self.state = P.State.STOPPED; self.busy = 0; self.leaving = False

async def main():
    global ok, fail
    settings = load(one_model(models={"m": {
        "script": str(SCRIPT), "cwd": str(ROOT / "model"), "vram_gb": 1,
        "port": 8100, "aliases": ["alias-m"]}}, internal_ports=[9500, 9600], admin_port=4998))

    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=9599, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    Wired.upstream_port = 9599

    async def probe(): return {0: P.Memory(80, 80)}
    cluster = P.Cluster(settings, probe=probe, factory=Wired, control=httpx.AsyncClient(timeout=10))
    proxy = P.Proxy(settings, cluster)
    front = uvicorn.Server(uvicorn.Config(proxy.application(8100), host="127.0.0.1",
                                          port=8100, log_level="error"))
    front_task = asyncio.create_task(front.serve())
    while not front.started:
        await asyncio.sleep(0.02)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8100", timeout=20) as c:
        r = await c.post("/v1/chat/completions?debug=1", json={"model": "alias-m", "hi": 1})
        check("POST дошёл", r.status_code == 200, r.text[:200])
        check("query проброшен в POST", seen["post"][1] == "debug=1", seen.get("post"))
        check("алиас переписан в каноническое имя (vLLM знает только его)",
              seen["post"][2]["model"] == "m", seen.get("post"))

        r = await c.get("/v1/models")
        check("/v1/models отвечает прокси из конфига",
              {m["id"] for m in r.json()["data"]} == {"m", "alias-m"}, r.text[:200])


        r = await c.post("/v1/../sleep", json={"model": "m"})
        check("обход по точкам не проходит", r.status_code == 404, f"{r.status_code} {r.text[:120]}")
        r = await c.get("/v1/chat/completions")
        check("GET к модели не проксируется", r.status_code == 405, r.status_code)

        got = []
        async with c.stream("POST", "/v1/chat/completions", json={"model": "m", "stream": True}) as r:
            check("стрим начался", r.status_code == 200)
            check("заголовки против буферизации", r.headers.get("x-accel-buffering") == "no")
            async for chunk in r.aiter_bytes():
                got.append(chunk)
        check("стрим дошёл целиком", b"[DONE]" in b"".join(got), b"".join(got)[:200])

        r = await c.post("/v1/chat/completions", json={"model": "m", "broken": True})
        check("битый JSON от модели не роняет прокси", r.status_code == 200, r.text[:200])

        await asyncio.sleep(0.2)
        inst = cluster.instances["m"]
        check("после всех запросов копия отпущена", inst.busy == 0, f"busy={inst.busy}")

    front.should_exit = server.should_exit = True
    await asyncio.gather(front_task, task, return_exceptions=True)
    await proxy.inference.aclose()
    await cluster.control.aclose()
    return report()

sys.exit(asyncio.run(main()))
