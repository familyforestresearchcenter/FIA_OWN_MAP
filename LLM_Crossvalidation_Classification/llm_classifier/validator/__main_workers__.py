# validator/dev/validators/__main__.py

import time
import csv
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Process, Queue

from .universal_classifier import UniversalClassifier
from .mp_workers import fast_worker, medium_worker


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Record:
    own1: str
    own2: str
    provided: int
    county: str
    state: str


@dataclass
class ClassificationResult:
    predicted: int
    provided: int
    correct: bool
    classifier_used: str
    needs_manual: bool


# =========================================================
# FIA → INTERNAL 1–7 MAPPING
# =========================================================

def remap_provided(fia_code: int) -> int:
    return {
        25: 1,
        31: 2,
        32: 3,
        41: 4,
        42: 5,
        43: 6,
        45: 7,
    }.get(fia_code, -1)


# =========================================================
# LOAD CSV RECORDS
# =========================================================

def load_records(csv_path: Path):
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                Record(
                    own1=row["OWN1"],
                    own2=row["OWN2"],
                    provided=int(row["Provided"]),
                    county=row["County"],
                    state=row["State"],
                )
            )
    return records


# =========================================================
# PIPELINE IMPLEMENTATION
# =========================================================

def run_pipeline():
    total_start = time.time()

    base = Path(__file__).resolve().parent.parent
    data_file = base / "data" / "training_data.csv"
    prompts_dir = base / "prompts"
    processed_dir = base / "processed"
    processed_dir.mkdir(exist_ok=True)

    output_file = processed_dir / "processed_results.csv"
    manual_file = processed_dir / "manual_review.csv"

    records = load_records(data_file)
    ordered_results = [None] * len(records)

    fast_prompt = (prompts_dir / "classifier_fast.txt").read_text(encoding="utf-8")
    medium_prompt = (prompts_dir / "classifier_medium.txt").read_text(encoding="utf-8")

    MODEL_DIR = Path(__file__).resolve().parents[3] / "phi-gpu" / "gpu" / "gpu-int4-awq-block-128"

    # =====================================================
    # FAST STAGE (2 WORKERS)
    # =====================================================

    print(f"\n--- Running FAST on {len(records)} records (2 workers) ---")
    t0 = time.time()

    fast_task_q = Queue()
    fast_result_q = Queue()

    fast_workers = [
        Process(target=fast_worker, args=(fast_task_q, fast_result_q, MODEL_DIR, fast_prompt))
        for _ in range(2)
    ]

    for w in fast_workers:
        w.start()

    for idx, record in enumerate(records):
        true_code = remap_provided(record.provided)
        fast_task_q.put((idx, record.own1, record.own2, true_code))

    for _ in fast_workers:
        fast_task_q.put(None)

    fast_failures = []

    for _ in range(len(records)):
        idx, pred, true_code = fast_result_q.get()
        record = records[idx]

        print(f"[FAST][IDX={idx}] True={true_code} | Pred={pred} | OWN1=\"{record.own1}\"")

        if pred == true_code and pred != -1:
            ordered_results[idx] = ClassificationResult(pred, true_code, True, "fast", False)
        else:
            fast_failures.append(idx)

    for w in fast_workers:
        w.join()

    print(
        f"[FAST] Completed. Correct={len(records) - len(fast_failures)}, "
        f"Failed={len(fast_failures)}, Time={round(time.time() - t0, 2)}s"
    )

    # =====================================================
    # MEDIUM STAGE (2 WORKERS)
    # =====================================================

    print(f"\n--- Running MEDIUM on {len(fast_failures)} records (2 workers) ---")
    t1 = time.time()

    medium_task_q = Queue()
    medium_result_q = Queue()

    medium_workers = [
        Process(target=medium_worker, args=(medium_task_q, medium_result_q, MODEL_DIR, medium_prompt))
        for _ in range(2)
    ]

    for w in medium_workers:
        w.start()

    for idx in fast_failures:
        record = records[idx]
        true_code = remap_provided(record.provided)
        medium_task_q.put((idx, record.own1, record.own2, true_code))

    for _ in medium_workers:
        medium_task_q.put(None)

    manual_review = []

    for _ in range(len(fast_failures)):
        idx, pred, true_code = medium_result_q.get()
        record = records[idx]

        print(f"[MEDIUM][IDX={idx}] True={true_code} | Pred={pred} | OWN1=\"{record.own1}\"")

        if pred == true_code and pred != -1:
            ordered_results[idx] = ClassificationResult(pred, true_code, True, "medium", False)
        else:
            ordered_results[idx] = ClassificationResult(pred, true_code, False, "medium", True)
            manual_review.append(record)

    for w in medium_workers:
        w.join()

    print(
        f"[MEDIUM] Completed. Correct={len(fast_failures) - len(manual_review)}, "
        f"Failed={len(manual_review)}, Time={round(time.time() - t1, 2)}s"
    )

    # =====================================================
    # WRITE OUTPUTS
    # =====================================================

    with output_file.open("w", encoding="utf-8", newline="") as outf, \
         manual_file.open("w", encoding="utf-8", newline="") as manf:

        outw = csv.writer(outf)
        manw = csv.writer(manf)

        outw.writerow(["OWN1", "OWN2", "Provided", "Predicted", "Correct", "Classifier"])
        manw.writerow(["OWN1", "OWN2", "Provided", "County", "State"])

        for rec, result in zip(records, ordered_results):
            outw.writerow([
                rec.own1,
                rec.own2,
                rec.provided,
                result.predicted,
                result.correct,
                result.classifier_used,
            ])

            if result.needs_manual:
                manw.writerow([
                    rec.own1,
                    rec.own2,
                    rec.provided,
                    rec.county,
                    rec.state,
                ])

    print(f"\nPipeline complete. Total Time: {round(time.time() - total_start, 2)}s")


if __name__ == "__main__":
    run_pipeline()
