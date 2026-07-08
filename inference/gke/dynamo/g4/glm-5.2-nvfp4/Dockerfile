# syntax=docker/dockerfile:1.7
# GLM-5.2-NVFP4 SM120 (RTX Pro 6000) SGLang runtime — the COMPLETE working recipe (see README.md):
#   1. sglang nightly base (transformers 5.12.1 -> parses glm_moe_dsa natively)
#   2. FlashInfer 0.6.13 (SM120 sparse-MLA kernels) + DeepGEMM 2.5.0 nv_dev (SM120 indexer), source-built
#   3. sglang PR #26928 BAKED IN (flashinfer_sparse_mla DSA backend; 4 files via `git apply --include`,
#      base_runner.py autotune hunks skipped: non-correctness, cutlass MoE already triggers autotune)
# Build context must contain pr26928.diff (this folder; refresh: `gh pr diff 26928 -R sgl-project/sglang`).
# Serve: --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
#   --kv-cache-dtype fp8_e4m3 + cutlass MoE x2 + --disable-shared-experts-fusion;
#   env SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0. Full flags in README.md.
# Notes: sglang cu13 image ships the CUDA toolkit at /usr/local/cuda (no pip-wheel nvrtc-link dance);
# sgl-deep-gemm is uninstalled so the real deep_gemm module wins.

ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260705-754524d8

FROM ${BASE_IMAGE} AS builder

USER root

ARG FLASHINFER_COMMIT=15a24591704fff8b9aedb9e3c6b2a9fa02fd9e22
ARG FLASHINFER_VERSION=0.6.13
ARG DEEPGEMM_COMMIT=a6b593d2826719dcf4892609af7b84ee23aaf32a
ARG MAX_JOBS=16

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/flashinfer-ai/flashinfer.git flashinfer \
    && git -C flashinfer checkout --detach "${FLASHINFER_COMMIT}" \
    && git -C flashinfer submodule update --init --depth 1 \
        3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl

RUN mkdir -p /wheelhouse \
    && python3 -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir /wheelhouse /build/flashinfer \
    && python3 -m pip download --no-deps --dest /wheelhouse \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && python3 -m pip download --no-deps --dest /wheelhouse \
        --index-url https://flashinfer.ai/whl/cu130/ \
        "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130"

RUN git clone --recursive --branch nv_dev https://github.com/deepseek-ai/DeepGEMM.git DeepGEMM \
    && git -C DeepGEMM checkout --detach "${DEEPGEMM_COMMIT}" \
    && git -C DeepGEMM submodule update --init --recursive

# sglang cu13 image has the CUDA toolkit at /usr/local/cuda -> plain build.
RUN set -eux; \
    cd /build/DeepGEMM; \
    rm -rf build dist ./*.egg-info; \
    CUDA_HOME=/usr/local/cuda \
    DG_FORCE_BUILD=1 \
    DG_USE_LOCAL_VERSION=1 \
    MAX_JOBS="${MAX_JOBS}" \
        python3 setup.py bdist_wheel; \
    cp dist/*.whl /wheelhouse/

FROM ${BASE_IMAGE}

USER root

COPY --from=builder /wheelhouse /tmp/wheelhouse

# Remove sglang's pinned deep_gemm packaging first so the new deep_gemm module wins,
# then force-install: flashinfer 0.6.13 (+cubin +jit-cache) and DeepGEMM 2.5.0.
RUN python3 -m pip uninstall -y sgl-deep-gemm deep-gemm 2>/dev/null; \
    python3 -m pip install --no-cache-dir --no-deps --upgrade --force-reinstall \
        /tmp/wheelhouse/*.whl \
    && rm -rf /tmp/wheelhouse

# sglang PR #26928: flashinfer_sparse_mla DSA backend (4 serving files; base_runner hunks skipped)
COPY pr26928.diff /tmp/pr26928.diff
RUN cd /sgl-workspace/sglang \
    && git config --global --add safe.directory /sgl-workspace/sglang \
    && git apply \
        --include='python/sglang/srt/layers/attention/dsa/flashinfer_sparse_mla.py' \
        --include='python/sglang/srt/layers/attention/dsa_backend.py' \
        --include='python/sglang/srt/server_args.py' \
        --include='python/sglang/srt/model_executor/pool_configurator.py' \
        /tmp/pr26928.diff \
    && python3 -m py_compile \
        python/sglang/srt/layers/attention/dsa/flashinfer_sparse_mla.py \
        python/sglang/srt/layers/attention/dsa_backend.py \
        python/sglang/srt/server_args.py \
        python/sglang/srt/model_executor/pool_configurator.py \
    && rm /tmp/pr26928.diff

# Preflight (off-GPU): wheel versions, deep_gemm symbols, PR backend registered, flashinfer API present.
RUN cd /tmp && python3 -P - <<'PY'
import deep_gemm, flashinfer, flashinfer_cubin, flashinfer_jit_cache
assert flashinfer.__version__ == "0.6.13", flashinfer.__version__
assert flashinfer_cubin.__version__ == "0.6.13"
assert flashinfer_jit_cache.__version__ == "0.6.13+cu130"
assert deep_gemm.__version__.startswith("2.5"), deep_gemm.__version__
for sym in ("get_paged_mqa_logits_metadata", "fp8_paged_mqa_logits",
            "fp8_mqa_logits", "get_num_sms"):
    assert hasattr(deep_gemm, sym), f"deep_gemm missing {sym}"
from sglang.srt.server_args import DSA_CHOICES
assert "flashinfer_sparse_mla" in DSA_CHOICES, DSA_CHOICES
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla  # noqa: F401
print("preflight ok:", flashinfer.__version__, deep_gemm.__version__, "| PR #26928 registered")
PY

# SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0 is REQUIRED for correctness on SM120:
# the dense-MHA short-prefill branch uses an SM100-only kernel that is silently incorrect.
ENV SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    FLASHINFER_WORKSPACE_BASE=/var/cache/glm52/flashinfer \
    DG_JIT_CACHE_DIR=/var/cache/glm52/deep_gemm \
    TRITON_CACHE_DIR=/var/cache/glm52/triton \
    XDG_CACHE_HOME=/var/cache/glm52/xdg \
    TORCHINDUCTOR_CACHE_DIR=/var/cache/glm52/torchinductor \
    CUDA_CACHE_PATH=/var/cache/glm52/cuda

RUN mkdir -p /var/cache/glm52/flashinfer /var/cache/glm52/deep_gemm \
        /var/cache/glm52/triton /var/cache/glm52/xdg \
        /var/cache/glm52/torchinductor /var/cache/glm52/cuda \
    && chmod -R 0777 /var/cache/glm52

LABEL org.opencontainers.image.title="GLM-5.2 NVFP4 SM120 SGLang runtime" \
      org.opencontainers.image.description="sglang nightly + FlashInfer 0.6.13 + DeepGEMM 2.5.0 nv_dev + sglang PR #26928 (flashinfer_sparse_mla SM120 DSA)" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang"

WORKDIR /sgl-workspace
EXPOSE 30000
STOPSIGNAL SIGTERM
