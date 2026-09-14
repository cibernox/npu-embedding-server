# npu-embedding-server

OpenAI-compatible embedding server for `Qwen/Qwen3-Embedding-0.6B` on Intel NPU, using bucketed static shapes (OpenVINO requires fixed input shapes on NPU).

## Pooling — read this before storing any vectors

Qwen3-Embedding is a causal model trained with **last-token pooling**: the embedding is the hidden state at the final position. This is not a stylistic choice — `Qwen/Qwen3-Embedding-0.6B/1_Pooling/config.json` sets `pooling_mode_lasttoken: true` and `pooling_mode_mean_tokens: false`.

Versions of this server before the `POOLING` env var **mean-pooled**, which has two consequences:

1. **Mean-pooled vectors are in a different vector space than every hosted Qwen3-Embedding endpoint.** You cannot embed documents here and queries at DeepInfra (or vice versa) — cosine similarity between the two is noise, and nothing raises an error. Retrieval just quietly returns the wrong things.
2. It scores below the model's published benchmarks, which are measured with the intended pooling.

Default is now `POOLING=last_token`. `POOLING=mean` exists only to read back an index built by an older deployment, or to A/B the two. **Changing this setting invalidates every stored vector — you must re-embed.**

`GET /health` reports the active pooling so a client can assert compatibility before trusting a shared index.

## Buckets

The bucket is chosen automatically as the smallest one that fits the longest input in the request. **The `model` field in the request body is ignored** — you cannot select a bucket through the API.

**Vectors are interchangeable across buckets.** Same weights, same pooling, and padding is masked out, so the same text embedded via the 64 and 512 buckets yields `cos = 1.000000`. The bucket changes only the compiled shape, never the vector space. Querying on a small bucket against documents ingested on a large one is therefore correct — and desirable, because it is ~5.6× faster.

So the intended layout is two buckets: a small one sized for queries, a large one for document chunks. With Qwen3-Embedding's 21-token instruction prefix, realistic search queries measure 28–42 tokens, which fits the 64 bucket with headroom.

Switching buckets is a cached NPU recompile costing **~800 ms**, paid by the request that triggers it. In practice this is immaterial: ingestion is a batch job that runs when content changes, so the cost lands twice per import rather than per request. `/health` exposes a `bucket_switches` counter so an unexpected thrash pattern is visible.

Any bucket size works — the directory suffix registers it. A query longer than the small bucket simply falls through to the next one up.

## Truncation is silent

Input longer than the active bucket is truncated, and the discarded content is simply gone — a 5,000-token document in a 512 bucket loses ~90% of itself. Chunk to fit before sending. The server logs a warning when it truncates, and restores the tokenizer's terminator token at the cut so last-token pooling still reads a meaningful position.

## Performance (Intel Core Ultra 7 265K, NPU 3.7, FP32, measured)

Measured on NPU with openvino 2026.3.1 + linux-npu-driver v1.38.0.

| Workload | Bucket | p50 | p95 |
|---|---|---|---|
| Search query, ~11 tok | 64 | **46 ms** | 118 ms |
| Search query padded to ~95 tok | 512 | 277 ms | 371 ms |
| Document chunk, ~405 tok | 512 | 314 ms | 363 ms |
| Batch of 4 chunks | 512 | 1,049 ms (262 ms/chunk) | |
| Batch of 16 chunks | 512 | 4,192 ms (262 ms/chunk) | |
| Batch of 32 chunks | 512 | 8,268 ms (258 ms/chunk) | |
| Bucket switch (cached) | — | ~1,000 ms | |
| Cold compile, empty cache | — | 8.2 s | |

Batching saves HTTP round trips but not compute — inputs are embedded in a loop, so per-chunk cost is flat at ~260 ms. Embedding a 677-chunk corpus takes **~3 minutes**.

The driver/OpenVINO bump was worth ~2-4% (queries 50 -> 46 ms, ingest 270 -> 262 ms/chunk) — parity rather than a speedup, so treat it as maintenance, not optimization.

### NPU vs CPU numerics

`tools/probe_compare.py` across 11 mixed probes: **min 0.999946, mean 0.999990**. The NPU is numerically equivalent to the CPU runner for practical purposes, so a CPU fallback can serve queries against an NPU-built index. Drift grows with input length (the longest probe is the 0.999946), which is why the probe set spans short queries through multi-hundred-token passages rather than testing one short string.

## Quick Start

```bash
docker run -d --name npu-embedding-server \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  -v /path/to/models:/models \
  -e API_KEY=your-secret \
  -p 8100:8100 \
  ghcr.io/cibernox/npu-embedding-server:latest
```

## Model Setup

The server expects one or more model directories in `/models` named `<model-name>-<bucket-size>`:

```
/models/
├── Qwen3-Embedding-0.6B-npu-64/     (queries)
├── Qwen3-Embedding-0.6B-npu-512/    (document chunks)
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   ├── tokenizer.json
│   └── ...
└── npu_cache/          (auto-created, stores compiled NPU blobs)
```

Export with `optimum-intel` (`pip install -r requirements-export.txt` — deliberately not in the serving image, since `server.py` never imports it):

```python
from optimum.intel import OVModelForFeatureExtraction

model = OVModelForFeatureExtraction.from_pretrained("Qwen/Qwen3-Embedding-0.6B", export=True, compile=False)

model.reshape(1, 64)
model.save_pretrained("/models/Qwen3-Embedding-0.6B-npu-64")

model.reshape(1, 512)
model.save_pretrained("/models/Qwen3-Embedding-0.6B-npu-512")
```

Any bucket size works — the directory suffix is what registers it. `reshape(1, 1024)` into `...-npu-1024` is a drop-in addition requiring no code change.

## Instruction prefixes are the client's job

The server tokenizes input verbatim; it applies no prompt template. Qwen3-Embedding is asymmetric — **queries** are meant to carry an instruction prefix, **documents** are embedded bare:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: how to prune tomato suckers
```

Prefixing changes the vector substantially (measured cosine ≈ 0.79 between prefixed and bare). Whatever you choose, apply it identically at index time and query time, and identically across any fallback provider.

## API

### POST /v1/embeddings

OpenAI-compatible. Accepts a string or an array. Bucket selection is automatic.

```bash
curl http://localhost:8100/v1/embeddings \
  -H 'Authorization: Bearer your-secret' \
  -H 'Content-Type: application/json' \
  -d '{"input": "how to prune tomato suckers"}'
```

Batching sends fewer HTTP round trips but does not batch on the NPU (inputs are embedded in a loop). Note that one long input drags the whole batch into a larger bucket.

`dimensions` and `encoding_format` are **not** implemented — requests including them are accepted and the parameters ignored. Output is always 1024 float32 values, L2-normalized. Qwen3-Embedding supports Matryoshka truncation, so a client may truncate and re-normalize itself.

### GET /health

Unauthenticated, so clients can health-check without credentials.

```json
{"status": "ok", "device": "NPU", "pooling": "last_token", "current_bucket": 64,
 "available_buckets": [64, 512], "bucket_switches": 0}
```

### GET /v1/models

Lists available bucket variants. Authenticated.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/models` | Directory containing bucket model dirs |
| `OPENVINO_DEVICE` | `NPU` | OpenVINO device (`NPU`, `GPU`, `CPU`) |
| `POOLING` | `last_token` | `last_token` (correct) or `mean` (legacy; different vector space) |
| `DEFAULT_BUCKET` | `64` | Bucket loaded at startup; falls back to the smallest available |
| `API_KEY` | _(unset)_ | Bearer token for `/v1/*`. **Unset means no auth** — only safe on a trusted LAN |
| `PORT` | `8100` | Server port |
| `NPU_CACHE_DIR` | `/models/npu_cache` | NPU compilation cache |

## Verifying vector-space compatibility

Before pointing a client at two endpoints — a local server and a hosted fallback, or two differently-configured deployments — confirm they agree:

```bash
tools/probe_compare.py http://npu-host:8100 https://api.deepinfra.com/v1/openai \
  --key-a "$NPU_KEY" --key-b "$DEEPINFRA_TOKEN" \
  --model-b Qwen/Qwen3-Embedding-0.6B --threshold 0.999
```

Exits non-zero if any probe falls short. A dimension mismatch fails immediately. Remember that agreement requires the same model *and* the same pooling *and* the same instruction prefixing.

**Verified pairing (2026-09-14):** this server vs **DeepInfra** `Qwen/Qwen3-Embedding-0.6B` — **min 0.999820, mean 0.999868: PASS**. Same 1024 dims, and ranking is identical (all three Willow regression queries at rank 1 through either endpoint). The residual ~0.00018 is precision, not pooling; a pooling mismatch scores ~0.6-0.8, not 0.9998. So the two can serve one index interchangeably.

Two DeepInfra defaults differ from this server and must be pinned when calling it:

- **`normalize` defaults to `false`** here it is always on. Their OpenAI-compatible route (`/v1/openai/embeddings`) does return L2-normalised vectors; their native route does not. Cosine is scale-invariant so ranking survives either way, but pgvector's inner-product operator (`<#>`) would silently break.
- **`custom_instruction`** lets them prepend Qwen's instruction template server-side. This server never templates, so leave it empty and prepend client-side, keeping both sides byte-identical.

Their URL convention also differs: the base already carries the version (`/v1/openai` + `/embeddings`), which is why `endpoint/1` in the tool branches on whether `/v1` is already present.

Latency for reference, measured from a Hetzner host: DeepInfra p50 206 ms / p95 414 ms (with occasional multi-second cold starts — one 7 s call in 45); this server over a Cloudflare tunnel p50 215 ms / p95 222 ms. Comparable medians; the self-hosted tail is tighter when the tunnel is healthy.

## Pinned dependencies

`requirements.txt` uses compatible-release pins rather than `>=` floors. The floors were a latent hazard: a rebuild would have silently pulled transformers 5.x, changing tokenizer behaviour under a server whose vectors must stay stable against a stored index. The NPU userspace driver and Level Zero loader are pinned in the `Dockerfile` for the same reason and should be bumped together, then re-tested on the NPU.

## How it works

1. On startup, scans `/models` for directories ending in `-<number>` containing `openvino_model.xml`
2. Loads `DEFAULT_BUCKET` and compiles it for NPU (first time ~12 s, cached ~1.4 s after)
3. On each request, tokenizes every input once, then picks the smallest bucket ≥ the longest input
4. If that bucket differs from the loaded one, releases the current model and compiles the new one (~800 ms cached)
5. Pools per `POOLING`, L2-normalizes, returns
