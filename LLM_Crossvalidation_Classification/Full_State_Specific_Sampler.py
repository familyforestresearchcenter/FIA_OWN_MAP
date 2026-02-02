import pandas as pd
import zipfile
import argparse
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VALID_OWNCDS = {25, 31, 32, 41, 42, 43, 45}
COUNTY_FIELD = "COUNTY_NAME"


def _norm_owner(s: pd.Series) -> pd.Series:
    """
    Normalize owner strings so duplicates collapse.
    """
    return (
        s.fillna("")
         .astype(str)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .str.upper()
    )

# ============================================================
# CORE LOGIC
# ============================================================

def extract_and_sample(
    state_abbr: str,
    round_dir: Path,
    n: int,
):
    zipped_dir = round_dir / "Zipped"
    extracted_dir = round_dir / "Extracted"
    samples_dir = round_dir / "Samples"

    samples_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Resolve parent state (TX from W_TX, etc.)
    # --------------------------------------------------
    parent_state = state_abbr.split("_")[-1]

    extract_path = extracted_dir / state_abbr
    csv_path = extract_path / f"{parent_state}_Full_Data_Table.csv"
    out_path = samples_dir / f"{parent_state}_OWNERSHIP_SAMPLE.csv"

    # --------------------------------------------------
    # Skip if already built (non-subregion only)
    # --------------------------------------------------
    if "_" not in state_abbr and out_path.exists():
        print(f"[SKIP] {parent_state} already sampled")
        return

    # --------------------------------------------------
    # Extract ZIP if needed
    # --------------------------------------------------
    zip_path = zipped_dir / f"{state_abbr}.zip"
    if not csv_path.exists():
        if not zip_path.exists():
            raise FileNotFoundError(
                f"No ZIP or extracted CSV found for {state_abbr}"
            )
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[EXTRACT] {state_abbr}")
    else:
        print(f"[EXTRACT] {state_abbr} already extracted")

    # --------------------------------------------------
    # Load + clean
    # --------------------------------------------------
    df = pd.read_csv(csv_path, low_memory=False)

    if COUNTY_FIELD not in df.columns:
        raise KeyError(f"Missing field: {COUNTY_FIELD}")

    df = df[df["OWNCD"].isin(VALID_OWNCDS)]

    # normalize BEFORE sampling
    df["OWN1"] = _norm_owner(df["OWN1"])
    df["OWN2"] = _norm_owner(df["OWN2"])

    df = df[(df["OWN1"] != "") | (df["OWN2"] != "")]

    # --------------------------------------------------
    # Sample N per OWNCD per STATE (NO COUNTY STRAT)
    # --------------------------------------------------
    sampled = (
        df
        .groupby("OWNCD", group_keys=False)
        .apply(lambda x: x.sample(n=min(n, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    sampled["STATE"] = parent_state

    sampled = sampled[
        ["OWN1", "OWN2", "OWNCD", COUNTY_FIELD, "STATE"]
    ]

    # --------------------------------------------------
    # Collapse sampled rows → unique ownerships
    # COUNT = frequency within sampled rows
    # --------------------------------------------------
    final = (
        sampled
        .groupby(
            ["OWN1", "OWN2", "OWNCD", COUNTY_FIELD, "STATE"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "COUNT"})
    )

    # --------------------------------------------------
    # Append if parent already exists (subregions)
    # --------------------------------------------------
    if out_path.exists():
        existing = pd.read_csv(out_path)
        final = (
            pd.concat([existing, final], ignore_index=True)
            .groupby(
                ["OWN1", "OWN2", "OWNCD", COUNTY_FIELD, "STATE"],
                as_index=False
            )["COUNT"]
            .sum()
        )

    # --------------------------------------------------
    # Sanity check (per OWNCD totals)
    # --------------------------------------------------
    totals = final.groupby("OWNCD")["COUNT"].sum()
    bad = totals[totals > n]
    if len(bad):
        print(f"[WARN] Some OWNCD totals exceed n={n}")
        print(bad)

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    final.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path} ({len(final)} rows)")


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate FIA ownership samples per OWNCD per STATE"
    )
    parser.add_argument(
        "round_dir",
        help="Round directory (e.g. Round6, Round7)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Samples per OWNCD per STATE (default: 100)"
    )
    parser.add_argument(
        "--states",
        nargs="*",
        help="Optional list of states or subregions (e.g. MI TX W_TX)"
    )

    args = parser.parse_args()

    round_dir = Path(args.round_dir).resolve()
    zipped_dir = round_dir / "Zipped"
    extracted_dir = round_dir / "Extracted"

    if not extracted_dir.exists():
        raise FileNotFoundError(f"Missing directory: {extracted_dir}")

    # --------------------------------------------------
    # Collect targets
    # --------------------------------------------------
    zip_targets = {f.stem for f in zipped_dir.glob("*.zip")} if zipped_dir.exists() else set()
    extracted_targets = {
        d.name for d in extracted_dir.iterdir()
        if d.is_dir() and (d / f"{d.name.split('_')[-1]}_Full_Data_Table.csv").exists()
    }

    all_targets = sorted(zip_targets | extracted_targets)

    if args.states:
        targets = [s for s in all_targets if s in set(args.states)]
    else:
        targets = all_targets

    if not targets:
        raise RuntimeError("No matching states found to process")

    print(f"[INFO] Target states: {targets}")

    for state_abbr in targets:
        try:
            extract_and_sample(
                state_abbr=state_abbr,
                round_dir=round_dir,
                n=args.n,
            )
        except Exception as e:
            print(f"[ERROR] {state_abbr}: {e}")


if __name__ == "__main__":
    main()
