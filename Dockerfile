FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Intel NPU userspace driver + Level Zero loader.
# Pinned deliberately: these two must be upgraded together and re-tested on the
# NPU, so they should never float. v1.38.0 lists Arrow Lake as a verified
# platform (Core Ultra 7 265K).
RUN curl -fL -O "https://github.com/oneapi-src/level-zero/releases/download/v1.33.1/libze1_1.33.1+u24.04_amd64.deb" && \
    apt-get update && apt-get install -y --no-install-recommends ./libze1*.deb && rm -f ./libze1*.deb && \
    curl -fL -o npu.tar.gz \
        "https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz" && \
    tar xzf npu.tar.gz && \
    apt-get install -y --no-install-recommends \
        ./intel-driver-compiler-npu_*.deb ./intel-fw-npu_*.deb ./intel-level-zero-npu_*.deb && \
    rm -f npu.tar.gz *.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY server.py /app/server.py

ENV MODELS_DIR=/models \
    OPENVINO_DEVICE=NPU \
    DEFAULT_BUCKET=64 \
    POOLING=last_token \
    PORT=8100 \
    METRICS_PORT=8101 \
    NPU_CACHE_DIR=/models/npu_cache

EXPOSE 8100 8101

CMD ["python3", "/app/server.py"]
