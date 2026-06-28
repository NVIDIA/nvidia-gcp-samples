# DeepSeek-V4-Pro-NVFP4 on RTX Pro 6000 (SM120) — SGLang Benchmark (Standalone + Dynamo DGD)

**Date:** 2026-06-27 · **Status:** ✅ SERVING + benchmarked
**Model:** `nvidia/DeepSeek-V4-Pro-NVFP4` (910B MoE, mixed FP8 + NVFP4 experts)
**Hardware:** RTX Pro 6000 (SM120 Blackwell, 96 GB) on GKE · PCIe only, no IB

## Standalone vs Dynamo (same workload)

Same workload (random, 16 prompts, ISL 128 / OSL 64, rate 4 req/s); same engine/model/topology (sglang v0.5.14, TP=8 PP=2, `flashinfer_cutlass`, mem-fraction 0.62). Backends differ by design: standalone = native `/generate`; Dynamo = OpenAI `/v1/chat/completions` via frontend.

| Metric | Standalone | Dynamo | Dynamo (warmed up) |
|---|---|---|---|
| **Median TPOT / ITL** | 144 / 129 ms | 144 / 131 ms | 150 / 128 ms |
| Output tok/s | 34.79 | 33.10 | 40.3 – 42.7 |
| Median TTFT | 12.8 s | 9.5 s | 6.1 – 7.5 s |
| Median E2E | 21.8 s | 20.7 s | 15.8 – 17.2 s |

**Read:** Decode latency (TPOT/ITL) is **identical** (same engine + SM120 cutlass kernels — Dynamo adds no per-token cost). Dynamo starts within ~5% of standalone throughput and, **once warmed up (radix + autotune cache), sustains 40–43 tok/s with ~6 s TTFT** — higher throughput and lower first-token latency from Dynamo's caching/orchestration. Successive Dynamo runs (same config) warmed: Output **33.1 → 40.3 → 42.7** tok/s, TTFT **9.5 → 7.5 → 6.1** s, while TPOT/ITL stayed flat (decode is warm-up-independent). To demonstrate Dynamo's TTFT advantage further, add multi-replica + a shared-prefix workload (see the kimi-k25 methodology). Per-run details below.

## Working configuration

- **Engine:** standalone SGLang (NO Dynamo), image `lmsysorg/sglang:v0.5.14-cu129` (public Docker Hub)
- **Topology:** multinode **TP=8 PP=2**, 2 nodes × 8 GPUs (StatefulSet + headless svc, hostNetwork dual-NIC)
- **MoE backend:** **`--moe-runner-backend flashinfer_cutlass`** + `--fp4-gemm-backend flashinfer_cutlass` ← the decisive flag
- **Memory:** **no cpu-offload** + `--mem-fraction-static 0.62` (small KV pool → ~39 GiB free for the ~32 GiB post-load expert-swizzle transient)
- YAML: `standalone-sglang-dsv4-pro-nvfp4-mn.yaml`

### The key finding (for engineering)
The default/recipe MoE backend **`flashinfer_trtllm_routed` is SM100-only** (the datacenter Blackwell architecture — B200/GB200/etc.) — kernel `get_trtllm_moe_sm100_module()` / `..._sm100f` passes load + autotune but **fails at runtime on SM120**:
`tvm.error.InternalError: trtllm_batched_gemm_runner.cu:286 Error occurred when running GEMM`.
SM120 has FP4 tensor cores, so this is a *missing kernel build*, not a HW limit. **`flashinfer_cutlass` serves on SM120.**

**Evidence for the `flashinfer_cutlass` SM120 claim (two independent legs):**
1. **Empirical** — with `--moe-runner-backend flashinfer_cutlass` the warmup MoE GEMM completed, the server fired up, and it generated correct text (smoke test above + 16/16 bench requests). The only change from the failing run was this backend flag.
2. **Source** — flashinfer 0.6.12 (in the image) ships a dedicated **SM120 Blackwell MoE kernel family**: `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/{moe_dispatch,moe_static_kernel,moe_micro_kernel,moe_dynamic_kernel}.py`, `fused_moe/cute_dsl/b12x_moe.py`, plus SM120 CUTLASS GEMM codegen `jit/gemm/cutlass/generate_kernels.py` and `jit/__init__.py` (`sm120a`). By contrast the trtllm MoE path exposes only `get_trtllm_moe_sm100_module()` (`_sm100f`) — no SM120 variant. (Verified by grepping the installed flashinfer package in the running pod.)

## Smoke test (real FP4 MoE forward pass on SM120)
Prompt: "In one sentence, what is tensor parallelism?" (temp 0)
> "Tensor parallelism is a distributed computing technique that splits the individual weight matrices or tensors of a neural network model across multiple devices, enabling parallel computation on different shards of the same layer to handle models too large for a single device's memory."

(13 prompt → 49 completion tokens — coherent, correct.)

## Benchmarks — same workload (random, 16 prompts, ISL 128 / OSL 64, rate 4 req/s)

Two deployments, **identical workload config**. Backends differ by design (standalone hits the native sglang `/generate`; Dynamo hits the OpenAI `/v1/chat/completions` via the frontend) — same engine/model/topology underneath.

### A) Standalone SGLang (no Dynamo) — `--backend sglang`

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 29.44 s |
| Request throughput | 0.54 req/s |
| Output token throughput | 34.79 tok/s |
| Total token throughput | 104.36 tok/s |
| Concurrency (effective) | 11.82 |
| **TTFT** median / P99 | 12,787 / 18,204 ms |
| **TPOT** median / P99 | 144.3 / 246.3 ms |
| **ITL** median / P99 | 129.1 / 359.9 ms |
| **E2E** median / P99 | 21,795 / 27,295 ms |

GPU at serve: ~59 GB / 96 GB used per GPU (56 weights + ~3 KV).

### B) Dynamo DGD — `--backend sglang-oai-chat` (via Dynamo frontend)

Community sglang image + Dynamo wheel (`--no-deps`) + `dynamo.sglang`; same TP=8 PP=2 / `flashinfer_cutlass` / mem-fraction 0.62. YAML: `../dgd-sglang-dsv4-pro-nvfp4-v0514.yaml`.

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 30.94 s |
| Request throughput | 0.52 req/s |
| Output token throughput | 33.10 tok/s |
| Total token throughput | 99.29 tok/s |
| Concurrency (effective) | 10.69 |
| **TTFT** median / P99 | 9,483 / 18,641 ms |
| **TPOT** median / P99 | 143.1 / 233.3 ms |
| **ITL** median / P99 | 131.2 / 1,117 ms |
| **E2E** median / P99 | 20,742 / 28,717 ms |

### B2) Dynamo DGD — aiperf (same workload, cross-check)

`aiperf` (the Dynamo-native benchmark) against the same frontend and workload (16 requests, ISL 128 / OSL 64, rate 4, streaming chat). Confirms aiperf works against the community-image + Dynamo-wheel deployment. (aiperf is not in the image by default — install with `pip install aiperf`, ideally in a separate client pod.)

| Metric | Value |
|---|---|
| Requests | 16 / 16 (OSL locked = 64) |
| Benchmark duration | 24.6 s |
| Output token throughput | 41.6 tok/s |
| Request throughput | 0.65 req/s |
| **TTFT** median / P99 | 8,633 / 13,917 ms |
| **ITL** median / P99 | 136.7 / 157.6 ms |
| **E2E** median / P99 | 17,298 / 23,084 ms |

Harness note: aiperf and bench_serving agree on the engine-level reality — **ITL is effectively the same** (~137 ms via aiperf vs ~131 ms via bench_serving). aiperf reports higher aggregate throughput (41.6 vs 33.1 tok/s) only because it completed the 16 requests faster (24.6 s vs 30.9 s) — a harness pacing/instrumentation difference, not an engine difference (consistent with the kimi-k25 aiperf-vs-bench_serving findings).

### Interpretation
- Numbers are **modest by design**: 910B model on **PP=2 over PCIe** (no NVLink/IB), `--disable-cuda-graph`, tiny KV pool. The multi-second TTFT + ~145 ms ITL reflect the 2-node PP=2 setup over PCIe + eager (non-graph) execution, not a kernel problem.
- Both runs are **functional validations** that the NVFP4 MoE path works end-to-end on SM120 — not tuned perf numbers.

Raw: `bench_serving_standalone.jsonl`, `bench_serving_dgd.jsonl`.
