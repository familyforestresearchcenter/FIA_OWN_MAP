# llm_classifier/validator/__main__.py

import time
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .universal_classifier import UniversalClassifier


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Record:
    own1: str
    own2: str
    provided: int          # FIA OWNCD (25–45)
    county: str | None
    state: str
    count: int             # sampling weight


# =========================================================
# FIA ↔ INTERNAL MAPPING
# =========================================================

FIA_TO_INTERNAL = {
    25: 1,
    31: 2,
    32: 3,
    41: 4,
    42: 5,
    43: 6,
    45: 7,
}

INTERNAL_TO_FIA = {v: k for k, v in FIA_TO_INTERNAL.items()}


def remap_provided(fia_code: int) -> int:
    return FIA_TO_INTERNAL.get(fia_code, -1)


def remap_pred(internal_code: int) -> Optional[int]:
    return INTERNAL_TO_FIA.get(internal_code)


# =========================================================
# LOAD SAMPLE RECORDS
# =========================================================

def load_records(csv_path: Path) -> list[Record]:
    records = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        has_county = "COUNTY_NAME" in reader.fieldnames

        for row in reader:
            records.append(
                Record(
                    own1=row["OWN1"],
                    own2=row["OWN2"],
                    provided=int(float(row["OWNCD"])),  # handles 41.0 safely
                    county=row["COUNTY_NAME"] if has_county else None,
                    state=row["STATE"],
                    count=int(row["COUNT"]),
                )
            )

    return records


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(
    samples_dir: Path,
    target_state: Optional[str] = None,
):
    total_start = time.time()

    BASE_DIR = Path(__file__).resolve().parent          # llm_classifier/validator
    LLM_ROOT = BASE_DIR.parent                          # llm_classifier

    PROMPTS_DIR = LLM_ROOT / "prompts"
    OUTPUT_DIR = LLM_ROOT / "outputs"
    MODEL_DIR = LLM_ROOT.parent / "phi-gpu" / "gpu" / "gpu-int4-awq-block-128"

    OUTPUT_DIR.mkdir(exist_ok=True)

    VERIFIED_PATH = OUTPUT_DIR / "verified.csv"
    UNVERIFIED_PATH = OUTPUT_DIR / "unverified.csv"

    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    # -----------------------------------------------------
    # Select sample files
    # -----------------------------------------------------
    sample_files = sorted(samples_dir.glob("*.csv"))

    if target_state:
        sample_files = [
            f for f in sample_files
            if f.name.startswith(f"{target_state}_")
        ]

    if not sample_files:
        raise RuntimeError(f"No sample files found for state '{target_state}'")

    # -----------------------------------------------------
    # Load prompts
    # -----------------------------------------------------
    fast_prompt = (PROMPTS_DIR / "classifier_fast.txt").read_text(encoding="utf-8")
    medium_prompt = (PROMPTS_DIR / "classifier_medium.txt").read_text(encoding="utf-8")

    FAST_CLF = UniversalClassifier(MODEL_DIR, fast_prompt, decode_steps=1)
    MEDIUM_CLF = UniversalClassifier(MODEL_DIR, medium_prompt, decode_steps=1)

    # -----------------------------------------------------
    # Prepare global outputs (append mode)
    # -----------------------------------------------------
    verified_exists = VERIFIED_PATH.exists()
    unverified_exists = UNVERIFIED_PATH.exists()

    verified_out = VERIFIED_PATH.open("a", encoding="utf-8", newline="")
    unverified_out = UNVERIFIED_PATH.open("a", encoding="utf-8", newline="")

    verified_writer = csv.writer(verified_out)
    unverified_writer = csv.writer(unverified_out)

    if not verified_exists:
        verified_writer.writerow([
            "OWN1", "OWN2", "OWNCD", "COUNTY", "STATE",
            "LLMPred", "LLMMatch", "TrueClass", "COUNT"
        ])

    if not unverified_exists:
        unverified_writer.writerow([
            "OWN1", "OWN2", "OWNCD", "COUNTY", "STATE",
            "LLMPred", "LLMMatch", "TrueClass", "COUNT"
        ])

    # =====================================================
    # PROCESS EACH SAMPLE FILE
    # =====================================================
    for csv_file in sample_files:
        print(f"\n=== Processing {csv_file.name} ===")
        records = load_records(csv_file)

        fast_failures: list[Record] = []
        verified_batch = []
        unverified_batch = []

        fast_start = time.time()

        # ---------------- FAST ----------------
        for record in records:
            true_internal = remap_provided(record.provided)
            pred_internal = FAST_CLF.classify(record)
            pred_fia = remap_pred(pred_internal)

            if pred_internal == true_internal and pred_fia is not None:
                verified_batch.append([
                    record.own1,
                    record.own2,
                    record.provided,
                    record.county,
                    record.state,
                    pred_fia,
                    True,
                    record.provided,
                    record.count,
                ])
            else:
                fast_failures.append(record)

        print(
            f"[FAST] Passed={len(records) - len(fast_failures)} "
            f"Failed={len(fast_failures)} "
            f"Time={round(time.time() - fast_start, 2)}s"
        )

        medium_start = time.time()

        # ---------------- MEDIUM ----------------
        for record in fast_failures:
            true_internal = remap_provided(record.provided)
            pred_internal = MEDIUM_CLF.classify(record)
            pred_fia = remap_pred(pred_internal)

            if pred_internal == true_internal and pred_fia is not None:
                verified_batch.append([
                    record.own1,
                    record.own2,
                    record.provided,
                    record.county,
                    record.state,
                    pred_fia,
                    True,
                    record.provided,
                    record.count,
                ])
            else:
                unverified_batch.append([
                    record.own1,
                    record.own2,
                    record.provided,
                    record.county,
                    record.state,
                    pred_fia,
                    False,
                    "",          # TrueClass intentionally blank
                    record.count,
                ])

        print(
            f"[MEDIUM] Passed={len(verified_batch) - (len(records) - len(fast_failures))} "
            f"Failed={len(unverified_batch)} "
            f"Time={round(time.time() - medium_start, 2)}s"
        )

        # ---------------- WRITE ONCE PER FILE ----------------
        verified_writer.writerows(verified_batch)
        unverified_writer.writerows(unverified_batch)

        print(
            f"[DONE] {csv_file.name} | "
            f"Verified={len(verified_batch)} | "
            f"Unverified={len(unverified_batch)}"
        )

    verified_out.close()
    unverified_out.close()

    print(f"\nPipeline complete in {round(time.time() - total_start, 2)}s")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM validator on ownership samples"
    )
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="Path to Samples directory"
    )
    parser.add_argument(
        "--state",
        help="Optional state or subregion to process (e.g. MI, TX, W_TX)"
    )

    args = parser.parse_args()

    run_pipeline(
        samples_dir=Path(args.samples_dir).resolve(),
        target_state=args.state,
    )
