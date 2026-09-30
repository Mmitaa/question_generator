"""Служебный порт. Ручка /health и аренда карт через HTTP."""
import asyncio
import sys

from fixture import FakeInstance, check, cluster, load, production, report
import proxy as P
import httpx
import uvicorn



async def main():
    settings = load(production())
    cl = cluster(settings)
    await cl.preload()
    proxy = P.Proxy(settings, cl)
    server = uvicorn.Server(uvicorn.Config(proxy.admin_application(), host="127.0.0.1",
                                           port=4997, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:4997", timeout=30) as c:
        r = await c.get("/health")
        check("/health отвечает", r.status_code == 200 and "gpus" in r.json(), r.text[:200])
        check("/health показывает модели", set(r.json()["models"]) == set(settings.models))

        r = await c.get("/admin/training")
        check("список аренд пуст", r.json() == {"leases": []}, r.text[:200])

        r = await c.post("/admin/training", json={"vram_gb": 14, "owner": "sft"})
        check("аренда по объёму создана", r.status_code == 200, r.text[:300])
        body = r.json()
        check("в ответе одна карта и free_gb", len(body["gpus"]) == 1 and body["free_gb"] > 13,
              body)
        lease_id = body["id"]

        r = await c.get("/admin/training")
        check("аренда видна в списке", [l["id"] for l in r.json()["leases"]] == [lease_id])

        r = await c.post("/admin/training", json={"vram_gb": 14, "gpus": [0]})
        check("gpus и vram_gb вместе -> 400", r.status_code == 400
              and "либо gpus" in r.json()["detail"], f"{r.status_code} {r.text[:200]}")
        r = await c.post("/admin/training", json={"vramgb": 14})
        check("опечатка в поле -> 400", r.status_code == 400
              and "непонятные поля" in r.json()["detail"], f"{r.status_code} {r.text[:200]}")
        r = await c.post("/admin/training", json={"vram_gb": 500})
        check("невыполнимый объём -> 400", r.status_code == 400, f"{r.status_code} {r.text[:200]}")

        r = await c.delete(f"/admin/training/{lease_id}")
        check("аренда снята", r.status_code == 200 and r.json()["id"] == lease_id, r.text[:200])
        r = await c.delete(f"/admin/training/{lease_id}")
        check("повторное снятие -> 404", r.status_code == 404, r.status_code)
        check("карты вернулись", not any(g.blocked for g in cl.gpus.values()))

    server.should_exit = True
    await task
    return report()

sys.exit(asyncio.run(main()))
