# Kimi K2.5 NVFP4 — KV-Aware Routing Benchmark

Demonstrates Dynamo's KV-aware routing advantage over standalone SGLang for shared-prefix workloads on memory-constrained hardware.

## Setup

| Component | Details |
|---|---|
| Model | nvidia/Kimi-K2.5-NVFP4 (MoE, NVFP4 quantized) |
| GPU | 2 nodes x 8x NVIDIA RTX PRO 6000 (96 GB each) |
| Parallelism | TP=8, DP-attention=8, no PP |
| Container | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0` |
| Memory | ~90.5 GB model weight/GPU, ~2.6 GB free for KV cache |
| KV dtype | fp8_e4m3 (MLA compressed, ~35 KB/token) |

## Server Configuration (identical for standalone and Dynamo)

```
--mem-fraction-static 0.97
--cuda-graph-max-bs 8
--max-running-requests 32
--context-length 16384
--tp-size 8
--dp-size 8
--enable-dp-attention
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--attention-backend flashinfer
--chunked-prefill-size 16384
--page-size 64
```

## Benchmark Parameters

| Parameter | Value | Rationale |
|---|---|---|
| ISL | ~1024 (820 prefix + 204 unique) | 80% shared-prefix workload |
| OSL | 8192 (forced via min/max tokens) | Long-form generation, stresses KV cache |
| Prefix pool | 32 distinct prompts | Enough diversity for cache pressure |
| Concurrency sweep | 4, 8, 16, 32 | Ramps load until OOM |
| Requests per point | 64 (profiling) + 32 (warmup) | Statistically meaningful |
| Seed | 100 (fixed) | Reproducible across runs |
| Tool | [aiperf](https://github.com/ai-dynamo/aiperf) v0.11.0 |

### aiperf command (per concurrency point)

```bash
aiperf profile \
  --model nvidia/Kimi-K2.5-NVFP4 \
  --url <endpoint> \
  --endpoint-type chat \
  --streaming \
  --tokenizer nvidia/Kimi-K2.5-NVFP4 \
  --tokenizer-trust-remote-code \
  --prefix-prompt-length 820 \
  --num-prefix-prompts 32 \
  --synthetic-input-tokens-mean 204 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8192 --output-tokens-stddev 0 \
  --num-prompts 64 --num-requests 64 \
  --warmup-request-count 32 \
  --use-server-token-count \
  --concurrency <4|8|16|32> \
  --record-processors 16 \
  --random-seed 100 \
  --extra-inputs ignore_eos:true \
  --extra-inputs max_tokens:8192 \
  --extra-inputs min_tokens:8192 \
  --extra-inputs temperature:0.0 \
  --extra-inputs repetition_penalty:1.0 \
  --artifact-dir <output-dir> \
  --ui simple
```

## Results

### Standalone SGLang (round-robin load balancing)

| Metric | Conc=4 | Conc=5 | Conc=6 | Conc=8 |
|---|---|---|---|---|
| Status | OK | OK | OOM/Crash | OOM/Crash |
| TTFT p50 (ms) | 448 | 653 | -- | -- |
| TTFT p90 (ms) | 634 | 823 | -- | -- |
| TTFT p99 (ms) | 701 | 942 | -- | -- |
| TTFT avg (ms) | 431 | 642 | -- | -- |
| ITL p50 (ms) | 30.9 | 33.5 | -- | -- |
| ITL avg (ms) | 30.3 | 33.2 | -- | -- |
| Output throughput (tok/s) | 129.2 | 132.3 | -- | -- |
| Request throughput (req/s) | 0.0158 | 0.0162 | -- | -- |
| Benchmark duration (s) | 4057 | 2971 | -- | -- |

### Dynamo with KV-Aware Routing

| Metric | Conc=4 | Conc=5 | Conc=8 | Conc=16 | Conc=32 |
|---|---|---|---|---|---|
| Status | OK | OK | OK | OK | OOM/Crash |
| TTFT p50 (ms) | 279 | 281 | 288 | 614 | -- |
| TTFT p90 (ms) | 363 | 363 | 525 | 1015 | -- |
| TTFT p99 (ms) | 510 | 635 | 793 | 1127 | -- |
| TTFT avg (ms) | 288 | 292 | 352 | 656 | -- |
| ITL p50 (ms) | 29.1 | 30.0 | 31.5 | 34.1 | -- |
| ITL avg (ms) | 29.3 | 29.7 | 31.3 | 34.1 | -- |
| Output throughput (tok/s) | 135.6 | 165.0 | 251.9 | 466.6 | -- |
| Request throughput (req/s) | 0.0165 | 0.0202 | 0.0308 | 0.0570 | -- |
| Per-user throughput (tok/s/user) | 34.2 | 33.0 | 31.9 | 29.3 | -- |
| Benchmark duration (s) | 3868 | 3178 | 2081 | 1124 | -- |

### Head-to-Head: Concurrency = 5

| Metric | Standalone | Dynamo | Improvement |
|---|---|---|---|
| TTFT p50 (ms) | 653 | 281 | **2.3x lower** |
| TTFT p90 (ms) | 823 | 363 | **2.3x lower** |
| TTFT avg (ms) | 642 | 292 | **2.2x lower** |
| ITL p50 (ms) | 33.5 | 30.0 | ~same |
| Output throughput (tok/s) | 132.3 | 165.0 | **+25%** |
| Max sustainable conc | **5** (OOM @ 6) | **16** | **3.2x higher** |

### TTFT Distribution (ms) — Conc=5

| Percentile | Standalone | Dynamo | Delta |
|---|---|---|---|
| p1 | 402 | 148 | ~similar (best-case cache hit) |
| p5 | 413 | 176 | **2.3x lower** |
| p10 | 434 | 201 | **2.2x lower** |
| p25 | 502 | 270 | **1.9x lower** |
| p50 | 653 | 281 | **2.3x lower** |
| p75 | 757 | 292 | **2.6x lower** |
| p90 | 823 | 363 | **2.3x lower** |
| p95 | 874 | 435 | **2.0x lower** |
| p99 | 942 | 635 | **1.5x lower** |

Standalone's p1 (402ms) is already **higher** than Dynamo's p50 (281ms) — even the luckiest standalone request is slower than the median Dynamo request. At the tail, Dynamo's p75 (292ms) is still lower than standalone's best case, showing that the vast majority of Dynamo requests benefit from prefix cache hits while standalone suffers from cache misses and KV pressure across the board.

### Full Comparison

| Metric | Standalone (conc=4) | Standalone (conc=5) | Dynamo (conc=4) | Dynamo (conc=5) | Dynamo (conc=8) | Dynamo (conc=16) |
|---|---|---|---|---|---|---|
| TTFT p50 (ms) | 448 | 653 | 279 (**-38%**) | 281 (**-57%** vs SA5) | 288 (**-56%** vs SA5) | 614 |
| TTFT p90 (ms) | 634 | 823 | 363 (**-43%**) | 363 (**-56%** vs SA5) | 525 (**-36%** vs SA5) | 1015 |
| ITL p50 (ms) | 30.9 | 33.5 | 29.1 | 30.0 | 31.5 | 34.1 |
| Output throughput (tok/s) | 129.2 | 132.3 | 135.6 | 165.0 (**+25%**) | 251.9 (**+90%**) | 466.6 (**+253%**) |
| Max sustainable conc | | **5** | | | | **16 (3.2x)** |

Note: At the same concurrency (5), Dynamo delivers 2.3x lower TTFT and 25% higher throughput. Standalone OOMs at conc=6 while Dynamo scales cleanly to conc=16.

## Key Findings

1. **Dynamo sustains 3.2x the concurrency** — standalone crashes at conc=6, Dynamo serves up to conc=16
2. **38% lower TTFT at equal load** — at conc=4, Dynamo TTFT p50=279ms vs standalone 448ms
3. **3.5x peak aggregate throughput** — 467 tok/s (Dynamo @ conc=16) vs 132 tok/s (standalone @ conc=5)
4. **Near-linear throughput scaling** — Dynamo scales 136 → 252 → 467 tok/s as concurrency increases from 4 → 8 → 16
5. **Standalone degrades sharply** — TTFT p50 jumps 46% (448 → 653ms) from conc=4 to conc=5, then OOMs at conc=6

## Why Dynamo Wins

### Standalone's hard ceiling: concurrency=4

Conc=8 crashed both standalone pods even on a fresh start (cold cache, no prior load). The standalone simply cannot handle 8 concurrent long-output requests with 32 prefixes in the KV pool.

The ~2.6 GB KV budget per GPU (~74K tokens) gets exhausted by:

| Component | Tokens |
|---|---|
| 32 cached prefixes x 820 tokens (radix cache) | ~26K |
| 4 active requests x ~10K tokens each (ISL + OSL) | ~40K |
| **Total at conc=4** | **~66K / 74K (89%)** |

At conc=8, each replica receives ~4 requests (round-robin across 2 replicas), but with all 32 prefixes cached the total demand jumps to ~80K+ tokens — exceeding the pool entirely and triggering OOM.

### Why Dynamo survives conc=8 and conc=16

With KV-aware routing, same-prefix requests are steered to the same replica. Each replica caches only ~16 prefixes (half the pool) instead of all 32:

| Component | Standalone (per replica) | Dynamo (per replica) |
|---|---|---|
| Cached prefixes | 32 x 820 = **26K tokens** | 16 x 820 = **13K tokens** |
| Savings from routing | 0 | **13K tokens freed** |
| Headroom for active requests | ~48K tokens | ~61K tokens |
| Max concurrent requests (10K each) | ~4 | ~6 |

This 13K-token savings translates to ~2 additional concurrent requests per replica. Combined with better cache hit rates (skip redundant prefill → lower TTFT), Dynamo can sustain conc=16 (8 per replica) because:
- Prefix cache hits mean active requests start generating immediately (no 820-token prefill allocation)
- Fewer total unique prefixes cached = less radix tree memory overhead
- The KV-aware router avoids "cache thrashing" where both replicas evict and re-cache the same prefixes

## Files

| File | Description |
|---|---|
| `standalone-sglang-kimi-k25-nvfp4.yaml` | Standalone SGLang Deployment (2 replicas, TP=8) |
| `dgd-kv-routing-experiment.yaml` | Dynamo DGD with KV-aware routing frontend |
| `benchmark-kimi-k25-nvfp4-pod.yaml` | aiperf client pod |
| `run-kv-routing-sweep.sh` | Concurrency sweep script |
| `results/standalone-conc4.json` | aiperf results — standalone @ concurrency=4 (initial run) |
| `results/standalone-conc4-run2.json` | aiperf results — standalone @ concurrency=4 (with warm cache) |
| `results/standalone-conc5.json` | aiperf results — standalone @ concurrency=5 |
| `results/dynamo-conc4.json` | aiperf results — Dynamo @ concurrency=4 |
| `results/dynamo-conc8.json` | aiperf results — Dynamo @ concurrency=8 |
| `results/dynamo-conc5.json` | aiperf results — Dynamo @ concurrency=5 |
| `results/dynamo-conc16.json` | aiperf results — Dynamo @ concurrency=16 |

## Reproduce

```bash
# 1. Deploy standalone
kubectl apply -f standalone-sglang-kimi-k25-nvfp4.yaml
kubectl rollout status deployment/standalone-sglang-kimi-k25-nvfp4 --timeout=900s

# 2. Deploy perf pod and copy script
kubectl apply -f benchmark-kimi-k25-nvfp4-pod.yaml
kubectl wait --for=condition=Ready pod/perf-kimi-k25-nvfp4 --timeout=300s
kubectl cp run-kv-routing-sweep.sh perf-kimi-k25-nvfp4:/workspace/

# 3. Run standalone sweep
kubectl exec perf-kimi-k25-nvfp4 -- /workspace/run-kv-routing-sweep.sh standalone

# 4. Swap to Dynamo
kubectl delete deployment standalone-sglang-kimi-k25-nvfp4
kubectl delete svc standalone-sglang-kimi-k25-frontend
kubectl apply -f dgd-kv-routing-experiment.yaml
kubectl get pods -l nvidia.com/dynamo-graph-deployment-name=agg-sglang-kimi-k25-nvfp4 -w

# 5. Run Dynamo sweep
kubectl exec perf-kimi-k25-nvfp4 -- /workspace/run-kv-routing-sweep.sh dynamo
```
