# validator/mp_workers.py

from types import SimpleNamespace
from .universal_classifier import UniversalClassifier


def fast_worker(task_q, result_q, model_dir, fast_prompt):
    clf = UniversalClassifier(model_dir, fast_prompt, decode_steps=1)

    while True:
        item = task_q.get()
        if item is None:
            break

        idx, own1, own2, true_code = item

        record = SimpleNamespace(
            own1=own1,
            own2=own2,
        )

        pred = clf.classify(record)
        result_q.put((idx, pred, true_code))


def medium_worker(task_q, result_q, model_dir, medium_prompt):
    clf = UniversalClassifier(model_dir, medium_prompt, decode_steps=1)

    while True:
        item = task_q.get()
        if item is None:
            break

        idx, own1, own2, true_code = item

        record = SimpleNamespace(
            own1=own1,
            own2=own2,
        )

        pred = clf.classify(record)
        result_q.put((idx, pred, true_code))
