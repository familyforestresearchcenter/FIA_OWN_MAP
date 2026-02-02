import time
import csv
from dataclasses import dataclass
from pathlib import Path

from .universal_classifier import UniversalClassifier


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
    count: int


@dataclass
class ClassificationResult:
    predicted: int
    provided: int
    correct: bool
    classifier_used: str
    needs_manual: bool
    count: int


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
# LOAD RECORDS FROM ONE SAMPLE CSV
# =========================================================

def load_records(csv_path: Path):
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                Record(
                    own1=row["OWN1"],
                    own2=row["OWN2"],
                    provided=int(row["OWNCD"]),
                    county=row["COUNTY_NAME"],
                    state=row["STATE"],
                    count=int(row["COUNT"]),
                )
            )
    return records


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline():
    total_start = time.time()

    BASE_DIR = Path(__file__).resolve().parent
    SAMPLES_DIR = BASE_DIR / "data" / "samples"
    PROMPTS_DIR = BASE_DIR / "prompts"
    PROCESSED_DIR = BASE_DIR / "processed"
    PROCESSED_DIR.mkdir(exist_ok=True)

    VERIFIED_CSV = PROCESSED_DIR / "verified.csv"
    UNVERIFIED_CSV = PROCESSED_DIR / "unverified.csv"

    # -----------------------------------------------------
    # Load prompts
    # -----------------------------------------------------
    fast_prompt = (PROMPTS_DIR / "classifier_fast.txt").read_text(encoding="utf-8")
    medium_prompt = (PROMPTS_DIR / "classifier_medium.txt").read_text(encoding="utf-8")

    # -----------------------------------------------------
    # Initialize classifiers
    # -----------------------------------------------------
    MODEL_DIR = BASE_DIR.parent / "phi-gpu" / "gpu" / "gpu-int4-awq-block-128"

    FAST_CLF = UniversalClassifier(MODEL_DIR, fast_prompt, decode_steps=1)
    MEDIUM_CLF = UniversalClassifier(MODEL_DIR, medium_prompt, decode_steps=1)

    # -----------------------------------------------------
    # Prepare output files (write headers once)
    # -----------------------------------------------------
    if not VERIFIED_CSV.exists():
        with VERIFIED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["OWN1", "OWN2", "OWNCD", "Predicted", "Classifier", "COUNT", "County", "State"]
            )

    if not UNVERIFIED_CSV.exists():
        with UNVERIFIED_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["OWN1", "OWN2", "OWNCD", "COUNT", "County", "State"]
            )

    # =====================================================
    # PROCESS EACH SAMPLE FILE
    # =====================================================
    for csv_file in sorted(SAMPLES_DIR.glob("*.csv")):
        print(f"\n=== Processing {csv_file.name} ===")
        records = load_records(csv_file)
        ordered_results = [None] * len(records)

        # ---------------- FAST ----------------
        fast_failures = []
        t0 = time.time()

        for idx, record in enumerate(records):
            true_code = remap_provided(record.provided)
            pred = FAST_CLF.classify(record)

            if pred == true_code and pred != -1:
                ordered_results[idx] = ClassificationResult(
                    pred, true_code, True, "fast", False, record.count
                )
            else:
                fast_failures.append((idx, record))

        print(
            f"[FAST] Correct={len(records) - len(fast_failures)} "
            f"Failed={len(fast_failures)} "
            f"Time={round(time.time() - t0, 2)}s"
        )

        # ---------------- MEDIUM ----------------
        manual_records = []
        t1 = time.time()

        for idx, record in fast_failures:
            true_code = remap_provided(record.provided)
            pred = MEDIUM_CLF.classify(record)

            if pred == true_code and pred != -1:
                ordered_results[idx] = ClassificationResult(
                    pred, true_code, True, "medium", False, record.count
                )
            else:
                ordered_results[idx] = ClassificationResult(
                    pred, true_code, False, "medium", True, record.count
                )
                manual_records.append(record)

        print(
            f"[MEDIUM] Correct={len(fast_failures) - len(manual_records)} "
            f"Failed={len(manual_records)} "
            f"Time={round(time.time() - t1, 2)}s"
        )

        # =====================================================
        # WRITE BATCH OUTPUTS (append)
        # =====================================================
        with VERIFIED_CSV.open("a", newline="", encoding="utf-8") as vf, \
             UNVERIFIED_CSV.open("a", newline="", encoding="utf-8") as uf:

            vwriter = csv.writer(vf)
            uwriter = csv.writer(uf)

            for rec, res in zip(records, ordered_results):
                if res.correct:
                    vwriter.writerow([
                        rec.own1,
                        rec.own2,
                        rec.provided,
                        res.predicted,
                        res.classifier_used,
                        rec.count,
                        rec.county,
                        rec.state,
                    ])
                elif res.needs_manual:
                    uwriter.writerow([
                        rec.own1,
                        rec.own2,
                        rec.provided,
                        rec.count,
                        rec.county,
                        rec.state,
                    ])

    print(f"\nPipeline complete. Total Time: {round(time.time() - total_start, 2)}s")


if __name__ == "__main__":
    run_pipeline()
