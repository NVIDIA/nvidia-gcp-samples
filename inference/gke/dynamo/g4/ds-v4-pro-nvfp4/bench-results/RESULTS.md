# DeepSeek-V4-Pro-NVFP4 on RTX Pro 6000 (SM120) — SGLang Benchmark (Standalone + Dynamo DGD)

**Date:** 2026-06-29 · **Status:** ✅ SERVING + benchmarked
**Model:** `nvidia/DeepSeek-V4-Pro-NVFP4` (910B MoE, mixed FP8 + NVFP4 experts)
**Hardware:** RTX Pro 6000 (SM120 Blackwell, 96 GB) on GKE · PCIe only, no IB

## Standalone vs Dynamo (same workload)

Same workload (random, 16 prompts, ISL 128 / OSL 64, rate 4 req/s); same engine/model/topology (sglang v0.5.14, TP=8 PP=2, `flashinfer_cutlass`, mem-fraction 0.62, **CUDA graphs enabled**). First (non-warmed) run. Backends differ by design: standalone = native `/generate`; Dynamo = OpenAI `/v1/chat/completions` via frontend.

| Metric | Standalone | Dynamo | Δ (Dynamo vs Standalone) |
|---|---|---|---|
| Output tok/s | 56.6 | 54.7 | −3% |
| Median TTFT | 4.7 s | 5.0 s | +6% |
| Median TPOT | 106 ms | 107 ms | +1% |
| Median ITL | 80.5 ms | 80.6 ms | ~0% |

**Read:** Decode latency (TPOT/ITL) is **identical** (same engine + SM120 cutlass kernels + CUDA graphs — Dynamo adds no per-token cost; ITL ~80 ms either way). Dynamo runs within ~3% of standalone throughput and ~6% on first-token latency — the cost of its frontend/router orchestration, not the engine. Per-run details below.

## Working configuration

- **Engine:** standalone SGLang (NO Dynamo), image `lmsysorg/sglang:v0.5.14-cu129` (public Docker Hub)
- **Topology:** multinode **TP=8 PP=2**, 2 nodes × 8 GPUs (StatefulSet + headless svc, hostNetwork dual-NIC)
- **MoE backend:** **`--moe-runner-backend flashinfer_cutlass`** + `--fp4-gemm-backend flashinfer_cutlass` ← the decisive flag
- **Memory:** **no cpu-offload** + `--mem-fraction-static 0.62` (small KV pool → ~39 GiB free for the ~32 GiB post-load expert-swizzle transient)
- **Decode:** CUDA graphs **enabled** (no `--disable-cuda-graph`) — capture costs only ~0.29 GB at this mem-fraction and cuts median ITL ~38% vs eager
- YAML: `standalone-sglang-dsv4-pro-nvfp4-mn.yaml`

### The key finding (for engineering)
The default/recipe MoE backend **`flashinfer_trtllm_routed` is SM100-only** (the datacenter Blackwell architecture — B200/GB200/etc.) — kernel `get_trtllm_moe_sm100_module()` / `..._sm100f` passes load + autotune but **fails at runtime on SM120**:
`tvm.error.InternalError: trtllm_batched_gemm_runner.cu:286 Error occurred when running GEMM`.
SM120 has FP4 tensor cores, so this is a *missing kernel build*, not a HW limit. **`flashinfer_cutlass` serves on SM120.**

**Evidence for the `flashinfer_cutlass` SM120 claim (two independent legs):**
1. **Empirical** — with `--moe-runner-backend flashinfer_cutlass` the warmup MoE GEMM completed, the server fired up, and it generated correct text (smoke test + 16/16 bench requests). The only change from the failing run was this backend flag.
2. **Source** — flashinfer 0.6.12 (in the image) ships a dedicated **SM120 Blackwell MoE kernel family**: `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/{moe_dispatch,moe_static_kernel,moe_micro_kernel,moe_dynamic_kernel}.py`, `fused_moe/cute_dsl/b12x_moe.py`, plus SM120 CUTLASS GEMM codegen `jit/gemm/cutlass/generate_kernels.py` and `jit/__init__.py` (`sm120a`). By contrast the trtllm MoE path exposes only `get_trtllm_moe_sm100_module()` (`_sm100f`) — no SM120 variant. (Verified by grepping the installed flashinfer package in the running pod.)

## Smoke test (real FP4 MoE forward pass on SM120)
Prompt: "In one sentence, what is tensor parallelism?" (temp 0)
> "Tensor parallelism is a distributed computing technique that splits the individual weight matrices or tensors of a neural network model across multiple devices, enabling parallel computation on different shards of the same layer to handle models too large for a single device's memory."

(13 prompt → 49 completion tokens — coherent, correct.)

## Benchmarks — same workload (random, 16 prompts, ISL 128 / OSL 64, rate 4 req/s)

Two deployments, **identical workload config**. Backends differ by design (standalone hits the native sglang `/generate`; Dynamo hits the OpenAI `/v1/chat/completions` via the frontend) — same engine/model/topology underneath. First (non-warmed) run; CUDA graphs enabled.

### A) Standalone SGLang (no Dynamo) — `--backend sglang`

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 18.08 s |
| Request throughput | 0.88 req/s |
| Output token throughput | 56.64 tok/s |
| Total token throughput | 169.91 tok/s |
| Concurrency (effective) | 9.97 |
| **TTFT** median / P99 | 4,745 / 9,087 ms |
| **TPOT** median / P99 | 106.5 / 130.6 ms |
| **ITL** median / P99 | 80.5 / 914 ms |
| **E2E** median / P99 | 11,325 / 15,922 ms |

GPU at serve: ~59 GB / 96 GB used per GPU (weights + small KV + CUDA-graph capture ~0.3 GB).

### B) Dynamo DGD — `--backend sglang-oai-chat` (via Dynamo frontend)

Community sglang image + Dynamo wheel (`--no-deps`) + `dynamo.sglang`; same TP=8 PP=2 / `flashinfer_cutlass` / mem-fraction 0.62 / CUDA graphs enabled. YAML: `../dgd-sglang-dsv4-pro-nvfp4-v0514.yaml`.

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 18.72 s |
| Request throughput | 0.85 req/s |
| Output token throughput | 54.70 tok/s |
| Total token throughput | 164.11 tok/s |
| Concurrency (effective) | 9.92 |
| **TTFT** median / P99 | 5,010 / 9,721 ms |
| **TPOT** median / P99 | 107.1 / 134.1 ms |
| **ITL** median / P99 | 80.6 / 948 ms |
| **E2E** median / P99 | 11,663 / 16,556 ms |

### Interpretation
- **Decode latency (TPOT/ITL) is identical** standalone vs Dynamo — the Dynamo frontend/router adds no per-token cost (ITL ~80 ms on both).
- With CUDA graphs enabled, median **ITL is ~80 ms** (vs ~129 ms eager — a ~38% decode improvement). The remaining multi-second TTFT reflects the 910B model on **PP=2 over PCIe** (no NVLink/IB) + a compact KV pool — a topology cost, not a kernel problem.
- Both runs are **functional validations** that the NVFP4 MoE path works end-to-end on SM120 — modest by design (2-node PP=2 / PCIe), not performance-tuned numbers.

Raw: standalone `bench_serving_standalone.jsonl`; Dynamo `bench_serving_dgd.jsonl`.
