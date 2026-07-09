# GLM-5.2-NVFP4 on GKE g4 (RTX PRO 6000 / SM120) — SGLang standalone + NVIDIA Dynamo

Functional-test recipe for serving
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) on GKE `g4-standard` nodes
(8× NVIDIA RTX PRO 6000 Blackwell, SM120) with SGLang — standalone, and behind
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) (aggregated serving).

- **Topology:** TP=8, single node (~433 GB checkpoint fits 8× 96 GB with FP8 KV cache headroom).
- **Contents:** `Dockerfile` (self-contained image build) · `pr26928.diff` ·
  `standalone-sglang-glm52-nvfp4.yaml` · `dgd-sglang-glm52-nvfp4.yaml` (Dynamo) ·
  `smoke-test.sh` (functional validation) · `bench-results/`.

## Why stock SGLang fails on SM120

GLM-5.2 uses DeepSeek Sparse Attention (DSA). On SM120, the stock paths dead-end
(see [sglang#26087](https://github.com/sgl-project/sglang/issues/26087)):

| Component | Stock behavior on SM120 |
|---|---|
| DSA indexer | calls DeepGEMM → `Unsupported architecture` (released wheels have no SM120 build) |
| DSA attention `trtllm` / `flashmla_*` | SM100-only kernels |
| DSA attention `tilelang` | bf16-only kernel; needs ~166 KB shared memory > SM120's ~101 KB |
| flashinfer ≤ 0.6.12 wheels | lack the SM120 sparse-MLA kernels |

## The fix (3 components)

1. **[sglang PR #26928](https://github.com/sgl-project/sglang/pull/26928)** — adds a
   `flashinfer_sparse_mla` DSA prefill+decode backend using FlashInfer's public
   `trtllm_batch_decode_with_kv_cache_mla` API. Until it merges, the `Dockerfile` here applies the
   4 serving-relevant files from the PR diff at build time.
2. **[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 0.6.13** (commit `15a2459`) — first
   release with the SM120 sparse-MLA kernels — plus `flashinfer-cubin==0.6.13` and
   `flashinfer-jit-cache==0.6.13+cu130`.
3. **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) `nv_dev` branch** (commit `a6b593d`) —
   SM120-capable; the stock SGLang DSA indexer then works natively.

Plus one required flag: `--disable-shared-experts-fusion` — the ModelOpt NVFP4 checkpoint keeps
`mlp.shared_experts` un-quantized; fusing them into packed-FP4 slots fails.

(The image also sets `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0` defensively. On this pinned
SGLang build it is a no-op: the dense-MHA short-prefill branch is already restricted to SM90/SM100 by
an architecture check in `dsa_backend.py`, so SM120 always uses the sparse path — equivalent for short
sequences, since DSA top-k 2048 ≥ sequence length.)

## Prerequisites

- A GKE cluster with a `g4-standard` node pool (8× RTX PRO 6000 per node) and `kubectl` context set
  to it. For cluster and Dynamo platform setup, refer to the official Dynamo GKE guide:
  https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/cloud-providers/gke/gke.md
- **1 node (8 GPUs)** per deployment path (TP=8, single node) — run one path at a time on the same
  node, or both in parallel on two nodes.
- Docker with BuildKit on x86_64 and a container registry the cluster can pull from.
- A `ReadWriteMany` PVC (e.g. Filestore) with ~450 GB free for the model checkpoint.
- Dynamo path only: the Dynamo platform (operator + etcd + NATS) installed in the cluster.

## 1. Build the image

Requires Docker with BuildKit on x86_64. From this folder:

```bash
export IMAGE=<your-registry>/glm52-sm120-sglang:sgl-fi0.6.13-dg2.5.0-pr26928
DOCKER_BUILDKIT=1 docker build --build-arg MAX_JOBS=16 -t "$IMAGE" -f Dockerfile .
docker push "$IMAGE"
# optional: refresh the PR diff first
#   gh pr diff 26928 --repo sgl-project/sglang > pr26928.diff
```

The build compiles FlashInfer 0.6.13 and DeepGEMM (`nv_dev`) wheels, applies the PR files, and runs
an off-GPU preflight (asserts wheel versions, DeepGEMM symbols, and that the `flashinfer_sparse_mla`
backend is registered).

## 2. Stage the model

~433 GB (47 safetensors shards) onto a `ReadWriteMany` PVC (e.g. Filestore).

```bash
pip install -U "huggingface_hub[cli]"
hf download nvidia/GLM-5.2-NVFP4 --local-dir /path/on/pvc/GLM-5.2-NVFP4
```

## 3. Deploy

Replace `<YOUR_IMAGE>` and `<YOUR_MODEL_PVC>` in the yamls, then:

```bash
# SGLang standalone (native server, port 30000)
kubectl apply -f standalone-sglang-glm52-nvfp4.yaml

# or NVIDIA Dynamo aggregated (OpenAI-compatible frontend, port 8000);
# requires the Dynamo platform installed: https://github.com/ai-dynamo/dynamo
kubectl apply -f dgd-sglang-glm52-nvfp4.yaml
```

First launch spends 10–30 min on weight load, JIT compilation, autotune, and CUDA-graph capture.
Persist the JIT caches (`/var/cache/glm52` in the image) on a volume for fast restarts.

## 4. Validate

```bash
./smoke-test.sh <host> 30000 generate   # standalone
./smoke-test.sh <host> 8000  chat       # Dynamo frontend
```

Four deterministic (temperature-0) checks — factual answers, arithmetic, a ~3600-token long-context
retrieval, and (chat mode) exact instruction following. All four must pass; garbled output indicates
a wrong kernel path — re-check the image build and the serve flags above.

## Functional benchmark

`sglang.bench_serving`, random dataset, 16 prompts, ISL 128 / OSL 64, request rate 4; warm runs
(JIT caches populated), default config (prefix caching on). Details in
[`bench-results/RESULTS.md`](bench-results/RESULTS.md).

| Metric | SGLang standalone | Dynamo (aggregated) | Δ |
|---|---|---|---|
| Output tok/s | 123.61 | 126.95 | +2.7% |
| Median TTFT (ms) | 120.9 | 122.8 | +1.6% |
| Median TPOT (ms) | 45.5 | 42.6 | −6.3% |
| Median ITL (ms) | 33.5 | 32.2 | −4.0% |

Functional validation numbers, not performance-tuned; deltas mainly reflect the different bench
protocols (native vs OpenAI-chat endpoint) — the takeaway is parity.

## Notes

- Do not enable MTP/speculative decoding on SM120.
- Use `flashinfer_cutlass` for the NVFP4 MoE on SM120: the default `flashinfer_trtllm_routed`
  kernel is SM100-only (`_sm100f`) and fails at runtime on SM120 (see the
  [`ds-v4-pro-nvfp4`](../ds-v4-pro-nvfp4) recipe for the detailed evidence).
- `--disable-custom-all-reduce` is intentional: RTX PRO 6000 has no NVLink.
- Prefix (radix) caching is enabled and validated on both serving paths — correctness and per-token
  latency are unchanged vs the cache-disabled configuration. Add `--disable-radix-cache` only for
  benchmarking, so repeated prompts do not hit the cache and skew numbers.
- Context length up to 196608 was validated (the FP8 KV pool at mem-fraction 0.9 holds ~470k
  tokens on 96 GB GPUs).

## References

- Model: https://huggingface.co/nvidia/GLM-5.2-NVFP4
- SGLang SM120 GLM support PR (the fix used here): https://github.com/sgl-project/sglang/pull/26928
- Earlier SM120 GLM DSA enablement PR (starting point of this investigation):
  https://github.com/sgl-project/sglang/pull/29586
- Original SM120 issue: https://github.com/sgl-project/sglang/issues/26087
- FlashInfer: https://github.com/flashinfer-ai/flashinfer · DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
