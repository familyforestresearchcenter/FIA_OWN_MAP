import os
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
print(BASE_DIR)

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "classifier_heavy.txt"

MODEL_DIR = BASE_DIR / "phi-gpu" / "gpu" / "gpu-int4-awq-block-128"
MODEL_PATH = MODEL_DIR / "model.onnx"


# ---------------------------------------------------------
# LOAD PROMPT
# ---------------------------------------------------------

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

system_text = load_text(PROMPT_FILE)


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

def tok(t: str):
    """Encode text and ALWAYS return int64."""
    return tokenizer(t, return_tensors="np")["input_ids"][0].astype(np.int64)


# ---------------------------------------------------------
# ONNX SESSION
# ---------------------------------------------------------

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)


# ---------------------------------------------------------
# PRETOKENIZED PREFIX
# ---------------------------------------------------------

SYSTEM_IDS = tok(system_text + "\n\n")
OWN1_IDS   = tok("OWN1: ")
OWN2_IDS   = tok("\nOWN2: ")
CODE_IDS   = tok("\nCODE: ")
ANSWER_IDS = tok("\nAnswer: ")


# ---------------------------------------------------------
# PREALLOCATED EMPTY KV CACHE
# ---------------------------------------------------------

EMPTY_KV = [
    (
        np.zeros((1, 32, 0, 96), dtype=np.float16),  # key
        np.zeros((1, 32, 0, 96), dtype=np.float16)   # value
    )
    for _ in range(32)
]


# ---------------------------------------------------------
# BUILD INPUT
# ---------------------------------------------------------

def build_input(o1: str, o2: str, code: int):
    t1 = tok(o1)
    t2 = tok(o2)
    t3 = tok(str(code))

    full_ids = np.concatenate([
        SYSTEM_IDS,
        OWN1_IDS, t1,
        OWN2_IDS, t2,
        CODE_IDS, t3,
        ANSWER_IDS
    ])

    return full_ids.astype(np.int64)[None, :]


# ---------------------------------------------------------
# CLASSIFY (HEAVY - OPTIMIZED)
# ---------------------------------------------------------

def classify_heavy(record):
    """
    Heavy classifier (optimized Tier-1):
    - Single decoding step (fastest configuration)
    - Returns int 1–7 if valid, otherwise -1
    """

    o1, o2, code = record.own1, record.own2, record.provided
    start_time = time.time()

    input_ids = build_input(o1, o2, code)
    seq_len = input_ids.shape[1]

    inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
    }

    # Reuse EMPTY_KV (no allocations)
    for layer in range(32):
        key, val = EMPTY_KV[layer]
        inputs[f"past_key_values.{layer}.key"] = key
        inputs[f"past_key_values.{layer}.value"] = val

    # One-step decoding
    ort_out = session.run(None, inputs)
    logits = ort_out[0]

    token_id = int(np.argmax(logits[0, -1]))
    decoded = tokenizer.decode([token_id]).strip()

    elapsed = round(time.time() - start_time, 4)
    print(f"[HEAVY] {elapsed}s → {decoded}")

    if decoded in {"1", "2", "3", "4", "5", "6", "7"}:
        return int(decoded)

    return -1
