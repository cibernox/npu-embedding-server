FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Intel NPU userspace driver + Level Zero loader
RUN curl -fL -O "https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze1_1.28.6+u24.04_amd64.deb" && \
    apt-get update && apt-get install -y --no-install-recommends ./libze1*.deb && rm -f ./libze1*.deb && \
    curl -fL -o npu.tar.gz \
        "https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz" && \
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
    PORT=8100 \
    NPU_CACHE_DIR=/models/npu_cache

EXPOSE 8100

CMD ["python3", "/app/server.py"]
