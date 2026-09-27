"""Neural sentence embeddings via ONNX Runtime.

Downloads a small pre-converted embedding model once (~90MB) from Hugging
Face, then runs it locally through ONNX Runtime. Uses the QNN (NPU)
execution provider automatically when available (Snapdragon + onnxruntime-qnn
installed); otherwise falls back to CPU.
"""
import numpy as np
import onnxruntime as ort

MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
MODEL_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"

_cache: dict = {}


def _load():
    if "session" in _cache:
        return _cache["session"], _cache["tokenizer"]

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    tokenizer_path = hf_hub_download(MODEL_REPO, TOKENIZER_FILE)

    available = ort.get_available_providers()
    providers = [p for p in ("QNNExecutionProvider", "CPUExecutionProvider") if p in available]
    session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])

    tokenizer = Tokenizer.from_file(tokenizer_path)

    _cache["session"] = session
    _cache["tokenizer"] = tokenizer
    return session, tokenizer


def _pad_batch(encodings, max_length: int = 256) -> tuple:
    """Pad/truncate a batch of encodings to equal length ourselves, so this
    doesn't depend on the tokenizers library's own padding API (which has
    changed across versions)."""
    trimmed = [(e.ids[:max_length], e.attention_mask[:max_length]) for e in encodings]
    batch_len = max((len(ids) for ids, _ in trimmed), default=0)

    input_ids = np.zeros((len(trimmed), batch_len), dtype=np.int64)
    attention_mask = np.zeros((len(trimmed), batch_len), dtype=np.int64)
    for i, (ids, mask) in enumerate(trimmed):
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(mask)] = mask
    return input_ids, attention_mask


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average token vectors, ignoring padding, then L2-normalize."""
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-9, None)


def embed(texts: list[str]) -> np.ndarray:
    """Return one L2-normalized embedding vector per input text."""
    session, tokenizer = _load()
    encodings = tokenizer.encode_batch(texts)
    input_ids, attention_mask = _pad_batch(encodings)

    input_names = {i.name for i in session.get_inputs()}
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, feed)
    output_names = [o.name for o in session.get_outputs()]
    idx = output_names.index("last_hidden_state") if "last_hidden_state" in output_names else 0

    return _mean_pool(outputs[idx], attention_mask)


def available() -> bool:
    """True if the embedding model can be loaded (downloaded or cached)."""
    try:
        _load()
        return True
    except Exception:
        return False
