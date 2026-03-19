from __future__ import annotations

import re
import pickle
import pandas as pd
from typing import Iterable, Dict


# ------------------------------------------------------------
# Regex helpers
# ------------------------------------------------------------

def contains_keyword(text: str, wrd_list):
    pattern = create_regex_pattern_from_list(wrd_list)
    return re.search(pattern, text or "") is not None

def create_regex_pattern_from_list(wrd_list):
    joined = "|".join(v.strip() for v in wrd_list if v)
    return re.compile(r"\b(?:%s)\b" % joined, flags=re.I)


def create_regex_pattern_from_dict(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Legacy helper: expands regex replacement dict.
    Used by preprocess_names in legacy engine.
    """
    return mapping


def apply_regex(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    """
    Apply regex search to OWN1 / OWN2 and return boolean DataFrame.
    """
    return pd.DataFrame({
        "OWN1": df["OWN1"].fillna("").str.contains(regex, regex=True, case=False),
        "OWN2": df["OWN2"].fillna("").str.contains(regex, regex=True, case=False),
    })


# ------------------------------------------------------------
# Parallelization shim (NO-OP for now)
# ------------------------------------------------------------

def parallelize_on_rows(df: pd.DataFrame, func):
    """
    Legacy-compatible no-op parallelization shim.

    The legacy pipeline can parallelize here, but for this
    module we intentionally run single-process.
    """
    return df.applymap(func)


# ------------------------------------------------------------
# Pickle loaders
# ------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path):
    # legacy alias
    return load_pickle(path)
