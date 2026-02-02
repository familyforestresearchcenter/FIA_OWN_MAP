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

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "classifier_medium.txt"

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

def tok(t):
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
# STATIC CACHES (Tier-1 speed optimization)
# ---------------------------------------------------------

# Prebuild empty KV cache once — 64 tensors instead of allocating all per call.
EMPTY_KV = []
for _ in range(32):
    EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))  # key
    EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))  # value

# Shared position-id buffer for decoding steps
POS0 = np.array([[0]], dtype=np.int64)

# ---------------------------------------------------------
# BUILD INPUT
# ---------------------------------------------------------

def build_input(o1, o2, code):
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
# CLASSIFY (MEDIUM)
# ---------------------------------------------------------

def classify_medium(record):
    """
    MEDIUM classifier:
    - Up to 2 decoding steps
    - Returns int 1–7
    - Returns -1 if invalid
    """
    o1, o2, code = record.own1, record.own2, record.provided

    start_time = time.time()

    input_ids = build_input(o1, o2, code)
    seq_len = input_ids.shape[1]

    # Build base inputs (KV added below)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
    }

    # Insert static empty KV
    idx = 0
    for layer in range(32):
        inputs[f"past_key_values.{layer}.key"]   = EMPTY_KV[idx]; idx += 1
        inputs[f"past_key_values.{layer}.value"] = EMPTY_KV[idx]; idx += 1

    output_digits = ""

    # Up to 2 auto-regressive steps
    for step in range(2):
        ort_out = session.run(None, inputs)
        logits = ort_out[0]

        token_id = int(np.argmax(logits[0, -1]))
        decoded = tokenizer.decode([token_id]).strip()

        if decoded not in {"1","2","3","4","5","6","7"}:
            break

        output_digits += decoded

        # Prepare next token (INT64 only)
        inputs["input_ids"] = np.array([[token_id]], dtype=np.int64)
        inputs["attention_mask"] = np.ones((1, 1), dtype=np.int64)
        inputs["position_ids"] = POS0

        # Update KV cache from model output (2 tensors per layer)
        out_idx = 1
        for layer in range(32):
            inputs[f"past_key_values.{layer}.key"]   = ort_out[out_idx]; out_idx += 1
            inputs[f"past_key_values.{layer}.value"] = ort_out[out_idx]; out_idx += 1

    elapsed = round(time.time() - start_time, 4)
    print(f"[MEDIUM] {elapsed}s → {output_digits or '?'}")

    if output_digits in {"1","2","3","4","5","6","7"}:
        return int(output_digits)

    return -1
