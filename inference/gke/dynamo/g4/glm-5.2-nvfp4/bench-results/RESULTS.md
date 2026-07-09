# GLM-5.2-NVFP4 on RTX PRO 6000 (SM120) — SGLang Benchmark (Standalone + Dynamo)

**Date:** 2026-07-08 · **Status:** ✅ serving + benchmarked (default config, prefix caching on)
**Model:** `nvidia/GLM-5.2-NVFP4` (MoE with DeepSeek Sparse Attention; NVFP4 experts, FP8 KV cache)
**Hardware:** 8× RTX PRO 6000 (SM120 Blackwell, 96 GB) — one GKE `g4-standard` node, TP=8, PCIe only

## Standalone vs Dynamo (same workload)

Same workload (`sglang.bench_serving`, random dataset, 16 prompts, ISL 128 / OSL 64, rate 4 req/s);
same engine/model/topology (recipe per [`../README.md`](../README.md)). Backends differ by design:
standalone = native `/generate` (:30000); Dynamo = OpenAI `/v1/chat/completions` via the frontend
(:8000). Warm runs (JIT/autotune caches populated); token counts identical on both paths
(1,274 in / 496 out).

| Metric | Standalone | Dynamo (aggregated) | Δ (Dynamo vs standalone) |
|---|---|---|---|
| Output tok/s | 123.61 | 126.95 | +2.7% |
| Median TTFT (ms) | 120.9 | 122.8 | +1.6% |
| Median TPOT (ms) | 45.5 | 42.6 | −6.3% |
| Median ITL (ms) | 33.5 | 32.2 | −4.0% |

**Read:** parity — Dynamo's frontend/router adds no measurable engine cost on a single aggregated
node. The strongest datapoint is the **fresh-prompt check** (first bench of each deployment — prompts
never seen before, so no prefix-cache benefit): standalone **112.88 tok/s / median TTFT 245 ms** vs
Dynamo **112.91 tok/s / 239 ms** — identical to 0.03%. The bench is seeded, so the reported (second)
run repeats the same prompts and its prefills partially hit the prefix cache — identically for both
columns — which is why its TTFT (~121 ms) is lower than the fresh-prompt TTFT (~240 ms).

## Per-run details

### A) SGLang standalone — `--backend sglang`

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 4.01 s |
| Request throughput | 3.99 req/s |
| Output token throughput | 123.61 tok/s (peak 198) |
| Total token throughput | 441.12 tok/s |
| Concurrency (effective) | 5.95 |
| **TTFT** median / P99 | 120.9 / 220.8 ms |
| **TPOT** median / P99 | 45.5 / 111.2 ms |
| **ITL** median / P99 | 33.5 / 138.8 ms |
| **E2E** median / P99 | 1,557 / 2,543 ms |

Fresh-prompt run (bench #1, no cache hits): 112.88 tok/s out, median TTFT 245.3 ms.
GPU at serve: ~88.5 / 97.9 GB used per GPU (weights + FP8 KV pool at `--mem-fraction-static 0.9` +
CUDA-graph capture).

### B) Dynamo DGD — `--backend sglang-oai-chat` (via Dynamo frontend)

Same image and engine flags wrapped in a `DynamoGraphDeployment` (frontend + one aggregated TP=8
worker); YAML: [`../dgd-sglang-glm52-nvfp4.yaml`](../dgd-sglang-glm52-nvfp4.yaml).

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 3.91 s |
| Request throughput | 4.10 req/s |
| Output token throughput | 126.95 tok/s (peak 190) |
| Total token throughput | 453.02 tok/s |
| Concurrency (effective) | 5.72 |
| **TTFT** median / P99 | 122.8 / 217.7 ms |
| **TPOT** median / P99 | 42.6 / 105.2 ms |
| **ITL** median / P99 | 32.2 / 135.4 ms |
| **E2E** median / P99 | 1,372 / 2,402 ms |

Fresh-prompt run (bench #1, no cache hits): 112.91 tok/s out, median TTFT 238.8 ms.
GPU at serve: ~88.4 / 97.9 GB used per GPU.

## Interpretation

- **Decode latency is effectively identical** standalone vs Dynamo (median ITL 33.5 vs 32.2 ms,
  TPOT 45.5 vs 42.6 ms) — the Dynamo frontend/router adds no per-token cost.
- **Fresh-prompt throughput is identical** (112.88 vs 112.91 tok/s), and fresh-prompt TTFT ~240 ms on
  both paths; repeated prefixes are served from the prefix cache (~121 ms TTFT), as designed.
- A cache-disabled A/B (`--disable-radix-cache`) showed unchanged correctness and per-token decode
  latency, so prefix caching on is the recommended configuration; disable it only for clean-prefill
  benchmarking.
- These are **functional validation** numbers on a single PCIe node, not performance-tuned: cold first
  launches additionally spend one bench on one-time JIT compilation (~29 tok/s, ~6 s TTFT) unless the
  JIT caches are persisted (see README).

Reproduce with the `sglang.bench_serving` command above — identical workload for both paths.
