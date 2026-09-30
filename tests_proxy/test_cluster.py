"""Размещение моделей по картам, гонки уборки и потолок ожидания."""
import asyncio
import sys
import time

from fixture import FakeInstance, check, cluster, layout, load, one_model, production, report
import proxy as P


async def race(cl, sleeper_call):
    """Ставит простаивающую копию под уборку и ровно в этот момент пытается её занять."""
    inst = await cl.acquire("m")
    inst.release()
    inst.last_used = time.monotonic() - 10_000
    FakeInstance.slow_probe = True

    async def grab():
        await asyncio.sleep(0.01)
        return cl.ready("m")

    async with cl.gpus[0].lock:
        _, grabbed = await asyncio.gather(sleeper_call(cl, inst, cl.gpus[0]), grab())
    FakeInstance.slow_probe = False
    return inst, grabbed


async def old_put_to_sleep(cl, instance, gpu):
    """Как было до правки. Вход закрывается уже после опроса карты."""
    await cl.free_now(gpu)
    await instance.set_sleeping(True)
    await cl.free_now(gpu)


async def main() -> int:
    one = load(one_model())

    print("\n[обычный путь]")
    cl = cluster(one, cards=1, card_gb=40)
    inst = await cl.acquire("m")
    check("копия поднялась и занята", inst.state is P.State.AWAKE and inst.busy == 1)
    inst.release()

    print("\n[гонка: занятую копию не усыпляют]")
    inst, grabbed = await race(cluster(one, cards=1, card_gb=40), old_put_to_sleep)
    check("старый вариант действительно ломался",
          grabbed is inst and inst.state is P.State.ASLEEP,
          "— иначе тест ничего не доказывает")
    cl = cluster(one, cards=1, card_gb=40)
    inst, grabbed = await race(cl, lambda c, i, g: c.reap_one(i, g))
    check("занятая и спящая одновременно невозможны",
          not (grabbed is inst and inst.state is P.State.ASLEEP))
    check("дверь закрылась синхронно, запрос уйдёт в очередь",
          grabbed is None and inst.state is P.State.ASLEEP and inst.busy == 0)

    print("\n[уборка свободной копии]")
    cl = cluster(one, cards=1, card_gb=40)
    inst = await cl.acquire("m")
    inst.release()
    inst.last_used = time.monotonic() - 10_000
    async with cl.gpus[0].lock:
        await cl.reap_one(inst, cl.gpus[0])
    check("свободная копия уснула", inst.state is P.State.ASLEEP and inst.slept == [True])
    check("и снова просыпается по запросу", (await cl.acquire("m")) is inst
          and inst.state is P.State.AWAKE)
    inst.release()

    print("\n[потолок ожидания]")
    cl = cluster(one, cards=1, card_gb=40)
    object.__setattr__(cl.settings, "queue_timeout", 0.3)
    object.__setattr__(cl.settings, "start_timeout", 0.3)
    await cl.model_locks["m"].acquire()
    started = time.monotonic()
    try:
        await cl.acquire("m")
        check("очередь ограничена", False, "— NoRoom не случился")
    except P.NoRoom as error:
        spent = time.monotonic() - started
        check("очередь ограничена wait_budget", 0.5 <= spent < 1.5 and "очереди" in str(error),
              f"— ждали {spent:.2f} сек: {error}")
    cl.model_locks["m"].release()

    print("\n[раскладка на двух картах по 16 ГБ]")
    cl = cluster(load(production()))
    await cl.preload()
    print(f"      {layout(cl)}")
    awake = {n for n, i in cl.instances.items() if i.state is P.State.AWAKE}
    check("активны ровно две модели, по одной на карту", len(awake) == 2, awake)
    check("приоритетная среди активных", "grade-ai" in awake, awake)
    check("остальные спят, а не остановлены",
          len(cl.instances) == 4 and all(i.state in (P.State.AWAKE, P.State.ASLEEP)
                                         for i in cl.instances.values()))
    switches = 0
    for _ in range(3):
        for name in ("grade-ai", "detector-14b"):
            before = cl.instances[name].state
            (await cl.acquire(name)).release()
            switches += cl.instances[name].state != before
    check("чередование запросов почти не двигает модели", switches <= 1, f"{switches} переключений")
    return report()


sys.exit(asyncio.run(main()))
