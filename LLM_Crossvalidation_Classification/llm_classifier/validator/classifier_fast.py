import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "classifier_fast.txt"

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

def tok(text: str):
    return tokenizer(text, return_tensors="np")["input_ids"][0].astype(np.int64)

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
# STATIC PRETOKENIZED PREFIX PARTS
# ---------------------------------------------------------

SYSTEM_IDS = tok(system_text + "\n\n")
OWN1_IDS   = tok("OWN1: ")
OWN2_IDS   = tok("\nOWN2: ")
ANSWER_IDS = tok("\nAnswer: ")

# ---------------------------------------------------------
# STATIC EMPTY KV CACHE
# ---------------------------------------------------------

EMPTY_KV = []
for _ in range(32):
    EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))
    EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))

# ---------------------------------------------------------
# BUILD INPUT (NO CODE FIELD)
# ---------------------------------------------------------

def build_input(o1, o2):
    parts = [
        SYSTEM_IDS,
        OWN1_IDS, tok(o1),
        OWN2_IDS, tok(o2),
        ANSWER_IDS,
    ]
    full = np.concatenate(parts, dtype=np.int64)
    return full[None, :]

# ---------------------------------------------------------
# CLASSIFY FAST
# ---------------------------------------------------------

def classify_fast(record):
    o1, o2 = record.own1, record.own2

    start = time.time()

    input_ids = build_input(o1, o2)
    seq_len = input_ids.shape[1]

    inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
    }

    idx = 0
    for layer in range(32):
        inputs[f"past_key_values.{layer}.key"]   = EMPTY_KV[idx]; idx += 1
        inputs[f"past_key_values.{layer}.value"] = EMPTY_KV[idx]; idx += 1

    ort_out = session.run(None, inputs)
    logits = ort_out[0]

    token_id = int(np.argmax(logits[0, -1]))
    decoded = tokenizer.decode([token_id]).strip()

    elapsed = round(time.time() - start, 4)
    print(f"[FAST] {elapsed}s → {decoded}")

    return int(decoded) if decoded in {"1","2","3","4","5","6","7"} else -1
