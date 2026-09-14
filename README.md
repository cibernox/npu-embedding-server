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

| Workload | Bucket | p50 |
|---|---|---|
| Search query, ~11 tok | 64 | **50 ms** |
| Search query padded to ~95 tok | 512 | 282 ms |
| Document chunk, ~405 tok | 512 | 289 ms |
| Batch of 4 chunks | 512 | 1,091 ms (273 ms/chunk) |
| Batch of 16 chunks | 512 | 4,325 ms (270 ms/chunk) |
| Batch of 32 chunks | 512 | 8,558 ms (267 ms/chunk) |
| Bucket switch (cached) | — | ~800 ms |
| First request after start (cold compile) | — | ~1.4 s |

Batching saves HTTP round trips but not compute — inputs are embedded in a loop, so per-chunk cost is flat at ~270 ms. Embedding a 677-chunk corpus takes ~3 minutes.

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

## Pinned dependencies

`requirements.txt` uses compatible-release pins rather than `>=` floors. The floors were a latent hazard: a rebuild would have silently pulled transformers 5.x, changing tokenizer behaviour under a server whose vectors must stay stable against a stored index. The NPU userspace driver and Level Zero loader are pinned in the `Dockerfile` for the same reason and should be bumped together, then re-tested on the NPU.

## How it works

1. On startup, scans `/models` for directories ending in `-<number>` containing `openvino_model.xml`
2. Loads `DEFAULT_BUCKET` and compiles it for NPU (first time ~12 s, cached ~1.4 s after)
3. On each request, tokenizes every input once, then picks the smallest bucket ≥ the longest input
4. If that bucket differs from the loaded one, releases the current model and compiles the new one (~800 ms cached)
5. Pools per `POOLING`, L2-normalizes, returns
