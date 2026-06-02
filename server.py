import gc
import os
import time
from typing import Union

import numpy as np
import openvino as ov
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DEVICE = os.environ.get("OPENVINO_DEVICE", "NPU")
DEFAULT_BUCKET = int(os.environ.get("DEFAULT_BUCKET", "64"))
PORT = int(os.environ.get("PORT", "8100"))
CACHE_DIR = os.environ.get("NPU_CACHE_DIR", f"{MODELS_DIR}/npu_cache")

app = FastAPI(title="NPU Embedding Server")
core = ov.Core()
tokenizer = None
buckets = {}
current_bucket = None
compiled_model = None


def discover_buckets():
    """Find all bucket model directories matching the naming convention."""
    found = {}
    for entry in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(path):
            continue
        xml_path = os.path.join(path, "openvino_model.xml")
        if not os.path.exists(xml_path):
            continue
        # Extract bucket size from directory name suffix (e.g., "model-name-64" -> 64)
        parts = entry.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            found[int(parts[1])] = xml_path
    return found


def find_tokenizer_path():
    """Find a directory containing tokenizer.json."""
    for entry in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, entry)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "tokenizer.json")):
            return path
    return None


def load_bucket(size):
    global current_bucket, compiled_model
    if current_bucket == size:
        return
    print(f"[embed] Switching to {size}-token bucket...", flush=True)
    start = time.time()
    if compiled_model is not None:
        del compiled_model
        compiled_model = None
        gc.collect()
    model = core.read_model(buckets[size])
    config = {"CACHE_DIR": CACHE_DIR} if DEVICE == "NPU" else {}
    compiled_model = core.compile_model(model, DEVICE, config=config)
    current_bucket = size
    print(f"[embed] Loaded {size}-token bucket in {time.time()-start:.1f}s", flush=True)


def select_bucket(token_count):
    for size in sorted(buckets.keys()):
        if token_count <= size:
            return size
    return max(buckets.keys())


def embed_text(text):
    enc = tokenizer(text, truncation=True, max_length=current_bucket, return_tensors="np")
    seq_len = enc["input_ids"].shape[1]
    input_ids = np.zeros((1, current_bucket), dtype=np.int64)
    attention_mask = np.zeros((1, current_bucket), dtype=np.int64)
    input_ids[0, :seq_len] = enc["input_ids"][0]
    attention_mask[0, :seq_len] = enc["attention_mask"][0]

    result = compiled_model({0: input_ids, 1: attention_mask})
    output = result[0]  # (1, seq_len, hidden_dim)
    mask = attention_mask[0, :, np.newaxis]
    pooled = (output[0] * mask).sum(axis=0) / mask.sum()
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist()


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "qwen3-embed"


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")

    max_tokens = 0
    for text in inputs:
        enc = tokenizer(text, truncation=False)
        max_tokens = max(max_tokens, len(enc["input_ids"]))

    bucket = select_bucket(max_tokens)
    load_bucket(bucket)

    embeddings = []
    total_tokens = 0
    for i, text in enumerate(inputs):
        enc = tokenizer(text, truncation=True, max_length=current_bucket)
        total_tokens += len(enc["input_ids"])
        vec = embed_text(text)
        embeddings.append({"object": "embedding", "index": i, "embedding": vec})

    return {
        "object": "list",
        "data": embeddings,
        "model": f"qwen3-embed-{DEVICE.lower()}-{current_bucket}",
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": f"qwen3-embed-{DEVICE.lower()}-{s}", "object": "model", "owned_by": "npu-embedding-server"}
            for s in sorted(buckets.keys())
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "current_bucket": current_bucket,
        "available_buckets": sorted(buckets.keys()),
    }


def main():
    global buckets, tokenizer

    buckets = discover_buckets()
    if not buckets:
        raise RuntimeError(f"No bucket models found in {MODELS_DIR}. Expected dirs ending in -<size> with openvino_model.xml")

    tok_path = find_tokenizer_path()
    if not tok_path:
        raise RuntimeError(f"No tokenizer found in {MODELS_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"[embed] Device: {DEVICE}", flush=True)
    print(f"[embed] Buckets: {sorted(buckets.keys())}", flush=True)
    print(f"[embed] Tokenizer: {tok_path}", flush=True)
    print(f"[embed] Cache: {CACHE_DIR}", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    load_bucket(DEFAULT_BUCKET if DEFAULT_BUCKET in buckets else min(buckets.keys()))
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
