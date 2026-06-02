# npu-embedding-server

OpenAI-compatible embedding server optimized for Intel NPU with bucketed static shapes.

Automatically selects the smallest compiled bucket that fits the input, minimizing latency for short queries (~30ms for typical search queries) while still handling full document chunks (~300ms for 512-token passages).

## Performance (Intel Core Ultra 7 265K, NPU 3.7)

| Input | Bucket | Latency |
|---|---|---|
| Search query (5-10 tokens) | 64 | **30ms** |
| Document chunk (400-512 tokens) | 512 | **264ms** |
| Bucket switch (cached) | — | ~1s one-time |

## Quick Start

```bash
docker run -d --name npu-embedding-server \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  -v /path/to/models:/models \
  -p 8100:8100 \
  ghcr.io/cibernox/npu-embedding-server:latest
```

## Model Setup

The server expects one or more model directories in `/models` following the naming convention `<model-name>-<bucket-size>`:

```
/models/
├── Qwen3-Embedding-0.6B-npu-64/
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   ├── tokenizer.json
│   └── ...
├── Qwen3-Embedding-0.6B-npu-512/
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   └── ...
└── npu_cache/          (auto-created, stores compiled NPU blobs)
```

Export models with `optimum-intel`:

```python
from optimum.intel import OVModelForFeatureExtraction

model = OVModelForFeatureExtraction.from_pretrained("Qwen/Qwen3-Embedding-0.6B", export=True, compile=False)

# 64-token bucket (for queries)
model.reshape(1, 64)
model.save_pretrained("/models/Qwen3-Embedding-0.6B-npu-64")

# 512-token bucket (for document chunks)
model.reshape(1, 512)
model.save_pretrained("/models/Qwen3-Embedding-0.6B-npu-512")
```

## API

### POST /v1/embeddings

OpenAI-compatible. Bucket selection is automatic based on input length.

```bash
curl http://localhost:8100/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "how to prune tomato suckers", "model": "qwen3-embed"}'
```

### GET /health

```json
{"status": "ok", "device": "NPU", "current_bucket": 64, "available_buckets": [64, 512]}
```

### GET /v1/models

Lists available bucket variants.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/models` | Directory containing bucket model dirs |
| `OPENVINO_DEVICE` | `NPU` | OpenVINO device (`NPU`, `GPU`, `CPU`) |
| `DEFAULT_BUCKET` | `64` | Bucket loaded at startup |
| `PORT` | `8100` | Server port |
| `NPU_CACHE_DIR` | `/models/npu_cache` | NPU compilation cache |

## Docker Compose

```yaml
services:
  npu-embedding-server:
    image: ghcr.io/cibernox/npu-embedding-server:latest
    container_name: npu-embedding-server
    restart: unless-stopped
    devices:
      - /dev/accel/accel0:/dev/accel/accel0
    ports:
      - "8100:8100"
    volumes:
      - /path/to/models:/models
    environment:
      - OPENVINO_DEVICE=NPU
      - DEFAULT_BUCKET=64
```

## How it works

1. On startup, scans `/models` for directories ending in `-<number>` containing `openvino_model.xml`
2. Loads the default bucket (64) and compiles it for NPU (first time ~12s, cached <1s after)
3. On each request, tokenizes input, picks the smallest bucket ≥ token count
4. If the needed bucket differs from the loaded one, releases the current model and loads the new one
5. The last-used bucket stays in memory until a different one is needed
