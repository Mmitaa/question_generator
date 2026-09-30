"""Аренда карт под обучение: по списку, по объёму и по умолчанию."""
import asyncio
import sys

from fixture import FakeInstance, check, cluster, load, production, report
import proxy as P




async def main():
    cl = cluster(load(production())); await cl.preload()
    lease = await cl.start_training(None, "sft", None, vram_gb=14.0)
    check("14 ГБ -> одна карта", len(lease.gpus) == 1, f"взяла {sorted(lease.gpus)}")
    check("взята карта без приоритетной grade-ai",
          not any(i.spec.priority for g in lease.gpus for i in cl.gpus[g].instances))
    inst = await cl.acquire("grade-ai")
    check("приоритетная работает на оставшейся карте", inst.state is P.State.AWAKE)
    inst.release(); await cl.end_training(lease.id)

    lease = await cl.start_training(None, "sft", None, vram_gb=25.0)
    check("25 ГБ -> обе карты", sorted(lease.gpus) == [0, 1], f"взяла {sorted(lease.gpus)}")
    check("все модели погашены", cl.instances == {})
    try:
        await cl.acquire("grade-ai"); check("во время обучения 503", False)
    except P.TrainingInProgress:
        check("во время обучения TrainingInProgress", True)
    await cl.end_training(lease.id)

    try:
        await cl.start_training(None, "sft", None, vram_gb=100.0)
        check("невыполнимый запрос отвергнут", False)
    except ValueError as e:
        check("невыполнимый запрос отвергнут с цифрами", "30.0 ГБ" in str(e), str(e))

    for gpus, need, needle in (([0], 5.0, "либо gpus, либо vram_gb"), ([], None, "gpus пуст"),
                               (None, -1.0, "больше нуля")):
        try:
            await cl.start_training(gpus, "x", None, need)
            check(f"gpus={gpus} vram_gb={need} отвергнут", False)
        except ValueError as e:
            check(f"gpus={gpus} vram_gb={need} отвергнут", needle in str(e), str(e))

    a = await cl.start_training(None, "sft", None, vram_gb=14.0)
    b = await cl.start_training(None, "eval", None, vram_gb=14.0)
    check("вторая аренда взяла другую карту", not (a.gpus & b.gpus), f"{a.gpus} и {b.gpus}")
    try:
        await cl.start_training(None, "third", None, vram_gb=14.0)
        check("третья аренда отвергнута", False)
    except ValueError as e:
        check("третья аренда отвергнута — карт больше нет", "0.0 ГБ" in str(e), str(e))
    await cl.end_training(a.id); await cl.end_training(b.id)

    lease = await cl.start_training([1], "по списку", None)
    check("список карт по-прежнему работает", lease.gpus == {1})
    await cl.end_training(lease.id)
    lease = await cl.start_training(None, "все", None)
    check("без полей забирает все карты", lease.gpus == {0, 1})
    await cl.end_training(lease.id)

    check("usable_gb учитывает резерв", abs(cl.usable_gb(cl.gpus[0]) - 15.0) < 0.01,
          cl.usable_gb(cl.gpus[0]))
    return report()

sys.exit(asyncio.run(main()))
