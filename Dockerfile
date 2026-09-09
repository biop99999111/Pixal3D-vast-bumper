# syntax=docker/dockerfile:1
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 ATTN_BACKEND=sdpa
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep expensive compilation cached when only application code changes.
COPY requirements.txt requirements-vast.txt ./
COPY configs/vast/source-refs.json ./configs/vast/source-refs.json
COPY scripts/setup_vast.sh scripts/natten_arch.py ./scripts/
ARG CUDA_ARCH_LIST=12.0
ARG BUILD_JOBS=4
RUN sed -i 's/\r$//' scripts/setup_vast.sh \
    && TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" NATTEN_CUDA_ARCH="${CUDA_ARCH_LIST}" \
       MAX_JOBS="${BUILD_JOBS}" NATTEN_N_WORKERS="${BUILD_JOBS}" \
       bash scripts/setup_vast.sh --build-only \
    && PIP_CONSTRAINT=/app/cache/build/torch-constraints.txt \
       python -m pip install 'jupyterlab>=4,<5' \
    && python -m pip check \
    && python -m ipykernel install --sys-prefix --name pixal3d-vast --display-name 'Pixal3D GPU' \
    && mkdir -p /opt/pixal3d-build \
    && cp cache/build-refs.json cache/requirements-resolved.txt /opt/pixal3d-build/ \
    && python -m pip freeze > /opt/pixal3d-build/image-requirements.txt \
    && printf '%s\n' "${CUDA_ARCH_LIST}" > /opt/pixal3d-build/cuda-architectures.txt \
    && rm -rf /app/cache /app/outputs /root/.cache/pip /app/cache-before-install.txt

COPY bumper_synth/ ./bumper_synth/
COPY pixal3d/ ./pixal3d/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY notebooks/ ./notebooks/
COPY assets/ ./assets/
COPY inference.py inference_mv.py app.py LICENSE NOTICE README.md ./

ENV HOME=/tmp \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    XDG_CACHE_HOME=/app/cache/xdg \
    TORCH_EXTENSIONS_DIR=/app/cache/torch_extensions \
    TRITON_CACHE_DIR=/app/cache/triton \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/app/cache/autotune_cache.json
RUN mkdir -p /app/inputs /app/outputs /app/cache \
    && chown -R 1000:1000 /app/inputs /app/outputs /app/cache /app/notebooks
USER 1000:1000
EXPOSE 8888
ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
