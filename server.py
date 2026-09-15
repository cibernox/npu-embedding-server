import gc
import os
import secrets
import time
from typing import Optional, Union

import numpy as np
import openvino as ov
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel
from transformers import AutoTokenizer

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DEVICE = os.environ.get("OPENVINO_DEVICE", "NPU")
DEFAULT_BUCKET = int(os.environ.get("DEFAULT_BUCKET", "64"))
PORT = int(os.environ.get("PORT", "8100"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8101"))
CACHE_DIR = os.environ.get("NPU_CACHE_DIR", f"{MODELS_DIR}/npu_cache")
API_KEY = os.environ.get("API_KEY", "").strip()

# Qwen3-Embedding is a causal model trained with last-token pooling: the vector
# is the hidden state at the final position, not an average over positions. See
# Qwen/Qwen3-Embedding-0.6B's 1_Pooling/config.json, which sets
# pooling_mode_lasttoken=true and pooling_mode_mean_tokens=false.
#
# Mean pooling produces vectors in a *different space* than every hosted
# Qwen3-Embedding endpoint, so a client cannot compare a locally-embedded
# document against a remotely-embedded query. It also scores below the model's
# published benchmarks, since those are measured with the intended pooling.
#
# "mean" is kept only so an existing index can be read back, or to A/B the two.
POOLING = os.environ.get("POOLING", "last_token").strip().lower()

app = FastAPI(title="NPU Embedding Server")
core = ov.Core()
tokenizer = None
terminator_id = None
buckets = {}
current_bucket = None
compiled_model = None
bucket_switches = 0

# Prometheus metrics on a dedicated port, not a route on the API port: the API
# port is exposed to the internet via a tunnel, and traffic metrics (request
# rates, token volumes) are usage telemetry, not something to publish. The
# metrics server serves only /metrics and is never published outside the LAN.
requests_total = Counter(
    "npu_embedding_requests_total",
    "HTTP requests by endpoint and status code.",
    ["endpoint", "status"],
)
request_duration_seconds = Histogram(
    "npu_embedding_request_duration_seconds",
    "Request latency in seconds.",
    ["endpoint"],
    # Buckets tuned to the measured NPU latency range (46 ms query through the
    # ~1 s bucket switch); the default scale is too coarse below 100 ms.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
)
tokens_total = Counter(
    "npu_embedding_tokens_total",
    "Input tokens fed to the model (post-truncation, pre-padding).",
    ["bucket"],
)
inputs_total = Counter(
    "npu_embedding_inputs_total",
    "Input texts embedded, summed over request batches.",
    ["bucket"],
)
truncated_inputs_total = Counter(
    "npu_embedding_truncated_inputs_total",
    "Input texts truncated to the bucket limit.",
)
bucket_switches_total = Counter(
    "npu_embedding_bucket_switches_total",
    "Bucket recompile switches.",
)
current_bucket_gauge = Gauge(
    "npu_embedding_current_bucket",
    "Bucket the compiled model currently serves.",
)
active_requests = Gauge(
    "npu_embedding_active_requests",
    "Requests currently in flight.",
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    endpoint = request.url.path
    active_requests.inc()
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        requests_total.labels(endpoint, "500").inc()
        raise
    finally:
        active_requests.dec()
    requests_total.labels(endpoint, str(response.status_code)).inc()
    request_duration_seconds.labels(endpoint).observe(time.monotonic() - start)
    return response


def require_auth(authorization: Optional[str] = Header(default=None)):
    """Bearer-token auth. Disabled when API_KEY is unset (LAN-only deployments)."""
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


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
    global current_bucket, compiled_model, bucket_switches
    if current_bucket == size:
        return
    # Vectors from different buckets are interchangeable (same weights, same
    # pooling, padding masked out — verified cos=1.000000), so serving short
    # queries from a small bucket and long chunks from a large one is the right
    # design: 50ms vs 282ms on a query. A switch is a cached NPU recompile of
    # ~800ms, which in practice costs nothing — ingestion is a batch job that
    # runs on content changes, so it is paid twice per import, not per request.
    # The counter is here purely so an unexpected thrash pattern is visible.
    if current_bucket is not None:
        bucket_switches += 1
        bucket_switches_total.inc()
        print(
            f"[embed] bucket switch {current_bucket} -> {size} "
            f"(total switches: {bucket_switches})",
            flush=True,
        )
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
    current_bucket_gauge.set(size)
    print(f"[embed] Loaded {size}-token bucket in {time.time()-start:.1f}s", flush=True)


def select_bucket(token_count):
    for size in sorted(buckets.keys()):
        if token_count <= size:
            return size
    return max(buckets.keys())


def embed_ids(ids):
    """Embed pre-tokenized ids. Returns (vector, tokens_used, was_truncated)."""
    n = current_bucket
    was_truncated = len(ids) > n
    if was_truncated:
        ids = ids[:n]
        # Last-token pooling reads the final position, so a truncated chunk
        # would otherwise be represented by whatever token the cut landed on.
        # Restore the terminator the tokenizer appends so the pooled position
        # keeps the meaning it was trained to carry.
        if terminator_id is not None:
            ids[-1] = terminator_id

    seq_len = len(ids)
    input_ids = np.zeros((1, n), dtype=np.int64)
    attention_mask = np.zeros((1, n), dtype=np.int64)
    input_ids[0, :seq_len] = ids
    attention_mask[0, :seq_len] = 1

    result = compiled_model({0: input_ids, 1: attention_mask})
    hidden = result[0][0]  # (bucket, hidden_dim)

    if POOLING == "mean":
        mask = attention_mask[0, :, np.newaxis]
        pooled = (hidden * mask).sum(axis=0) / mask.sum()
    else:
        pooled = hidden[seq_len - 1]

    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.tolist(), seq_len, was_truncated


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "qwen3-embed"


@app.post("/v1/embeddings", dependencies=[Depends(require_auth)])
def create_embeddings(req: EmbeddingRequest):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")

    # Tokenize once per input. The bucket is chosen from the longest input in
    # the batch, so a single long text drags the whole batch into a big bucket.
    encoded = [tokenizer(text, truncation=False)["input_ids"] for text in inputs]
    load_bucket(select_bucket(max(len(ids) for ids in encoded)))

    embeddings = []
    total_tokens = 0
    truncated = 0
    for i, ids in enumerate(encoded):
        vec, used, was_truncated = embed_ids(ids)
        total_tokens += used
        truncated += 1 if was_truncated else 0
        embeddings.append({"object": "embedding", "index": i, "embedding": vec})

    if truncated:
        # Truncation silently discards content — a 5k-token document embedded in
        # a 512 bucket loses ~90% of itself with no error. Callers should chunk
        # to fit; this at least leaves evidence when they haven't.
        print(
            f"[embed] WARNING {truncated}/{len(inputs)} input(s) truncated to "
            f"{current_bucket} tokens — content beyond the limit was discarded",
            flush=True,
        )

    tokens_total.labels(str(current_bucket)).inc(total_tokens)
    inputs_total.labels(str(current_bucket)).inc(len(inputs))
    truncated_inputs_total.inc(truncated)

    return {
        "object": "list",
        "data": embeddings,
        "model": f"qwen3-embed-{DEVICE.lower()}-{current_bucket}",
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
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
    # Unauthenticated on purpose: clients poll this to decide whether to route
    # here or to a fallback provider, and it exposes nothing sensitive.
    # `pooling` is reported so a client can assert vector-space compatibility
    # before trusting a shared index.
    return {
        "status": "ok",
        "device": DEVICE,
        "pooling": POOLING,
        "current_bucket": current_bucket,
        "available_buckets": sorted(buckets.keys()),
        "bucket_switches": bucket_switches,
    }


def main():
    global buckets, tokenizer, terminator_id

    if POOLING not in ("last_token", "mean"):
        raise RuntimeError(f"POOLING must be 'last_token' or 'mean', got {POOLING!r}")

    buckets = discover_buckets()
    if not buckets:
        raise RuntimeError(f"No bucket models found in {MODELS_DIR}. Expected dirs ending in -<size> with openvino_model.xml")

    tok_path = find_tokenizer_path()
    if not tok_path:
        raise RuntimeError(f"No tokenizer found in {MODELS_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # The token the tokenizer appends to every sequence — <|endoftext|> for
    # Qwen3-Embedding, which is what last-token pooling is meant to read.
    probe = tokenizer("probe", truncation=False)["input_ids"]
    terminator_id = probe[-1] if probe else None

    print(f"[embed] Device: {DEVICE}", flush=True)
    print(f"[embed] Pooling: {POOLING}", flush=True)
    print(f"[embed] Buckets: {sorted(buckets.keys())}", flush=True)
    print(f"[embed] Tokenizer: {tok_path} (terminator id: {terminator_id})", flush=True)
    print(f"[embed] Auth: {'enabled' if API_KEY else 'DISABLED'}", flush=True)
    print(f"[embed] Cache: {CACHE_DIR}", flush=True)

    if len(buckets) > 1:
        print(
            f"[embed] {len(buckets)} buckets present: {sorted(buckets.keys())}. "
            f"Vectors are interchangeable across buckets, so serve queries from the "
            f"small one and ingest from the large one.",
            flush=True,
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    load_bucket(DEFAULT_BUCKET if DEFAULT_BUCKET in buckets else min(buckets.keys()))
    start_http_server(METRICS_PORT, addr="0.0.0.0")
    print(f"[embed] Metrics: :{METRICS_PORT}/metrics (LAN only — never tunnel it)", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
