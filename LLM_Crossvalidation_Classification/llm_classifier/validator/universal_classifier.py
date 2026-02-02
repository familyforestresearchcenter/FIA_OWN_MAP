# universal_classifier.py

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer


class UniversalClassifier:
    """
    Unified ONNX classifier with:
    - Prompt injection (OWN1 + OWN2 only)
    - Single-step deterministic decoding
    - Correct KV-cache reuse (critical for Phi models)
    - Backward-compatible tokenizer handling
    """

    def __init__(
        self,
        model_dir,
        prompt_text,
        decode_steps=1,
        tokenizer_dir=None,   # ← OPTIONAL, backward-compatible
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.onnx"
        self.decode_steps = decode_steps

        # -------------------------
        # Prompt
        # -------------------------
        self.system_prompt = prompt_text.strip()

        # -------------------------
        # Tokenizer path resolution
        # -------------------------
        # If tokenizer_dir not provided, fall back to model_dir (original behavior)
        if tokenizer_dir is None:
            tokenizer_dir = self.model_dir
        else:
            tokenizer_dir = Path(tokenizer_dir)

        # -------------------------
        # Tokenizer (LOCAL ONLY)
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            trust_remote_code=True,
            local_files_only=True,
        )

        # -------------------------
        # ONNX Runtime session
        # -------------------------
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # -------------------------
        # Pre-tokenized prefix
        # -------------------------
        self.SYSTEM_IDS = self._tok(self.system_prompt + "\n\n")
        self.OWN1_IDS = self._tok("OWN1: ")
        self.OWN2_IDS = self._tok("\nOWN2: ")
        self.ANSWER_IDS = self._tok("\nAnswer: ")

        # -------------------------
        # Static empty KV template
        # Phi-3.5: 32 layers × (key,value)
        # -------------------------
        self.EMPTY_KV = []
        for _ in range(32):
            self.EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))  # key
            self.EMPTY_KV.append(np.zeros((1, 32, 0, 96), dtype=np.float16))  # value

        self.POS0 = np.array([[0]], dtype=np.int64)

    # ----------------------------------------------------
    def _tok(self, text):
        return self.tokenizer(text, return_tensors="np")["input_ids"][0].astype(np.int64)

    # ----------------------------------------------------
    def _build_prompt(self, record):
        o1 = record.own1 or ""
        o2 = record.own2 or ""

        full = np.concatenate(
            [
                self.SYSTEM_IDS,
                self.OWN1_IDS,
                self._tok(o1),
                self.OWN2_IDS,
                self._tok(o2),
                self.ANSWER_IDS,
            ],
            dtype=np.int64,
        )

        return full[None, :]

    # ----------------------------------------------------
    @staticmethod
    def _extract_digit(decoded_tokens):
        for t in decoded_tokens:
            t = t.strip()
            if t in {"1", "2", "3", "4", "5", "6", "7"}:
                return int(t)
        return -1

    # ----------------------------------------------------
    def classify(self, record):
        t0 = time.time()

        input_ids = self._build_prompt(record)
        seq_len = input_ids.shape[1]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
        }

        idx = 0
        for layer in range(32):
            inputs[f"past_key_values.{layer}.key"] = self.EMPTY_KV[idx]; idx += 1
            inputs[f"past_key_values.{layer}.value"] = self.EMPTY_KV[idx]; idx += 1

        decoded_tokens = []

        for _ in range(self.decode_steps):
            out = self.session.run(None, inputs)
            logits = out[0]

            tok_id = int(np.argmax(logits[0, -1]))
            tok_text = self.tokenizer.decode([tok_id]).strip()
            decoded_tokens.append(tok_text)

            inputs = {
                "input_ids": np.array([[tok_id]], dtype=np.int64),
                "attention_mask": np.ones((1, 1), dtype=np.int64),
                "position_ids": self.POS0,
            }

            out_idx = 1
            for layer in range(32):
                inputs[f"past_key_values.{layer}.key"] = out[out_idx]; out_idx += 1
                inputs[f"past_key_values.{layer}.value"] = out[out_idx]; out_idx += 1

        pred = self._extract_digit(decoded_tokens)

        print(
            f"[UNIV-{self.decode_steps}step] "
            f"{round(time.time() - t0, 4)}s → tokens={decoded_tokens} → pred={pred}"
        )

        return pred
