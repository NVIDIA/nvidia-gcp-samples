# Dynamo on GCP g4 (RTX PRO 6000 Blackwell) — Kimi K2.5 NVFP4

NVIDIA Dynamo + SGLang reference deployment for **Kimi K2.5 NVFP4** on Google Cloud's `g4-standard-384` instance (8× RTX PRO 6000 Blackwell, SM_120). 2-node GKE topology, TP=8 + PP=2.

## What's here

| File | Purpose |
|---|---|
| `standalone-sglang-kimi-k25-nvfp4.yaml` | Bare SGLang StatefulSet (2 nodes, TP=8 + PP=2) — matches Google's published Kimi NVFP4 reference flag-for-flag, but pins SGLang to `v0.5.10.post1` (the version Dynamo's `sglang-runtime:1.1.0` image bundles) for apples-to-apples comparison |
| `dgd-agg-sglang-kimi-k25-nvfp4-parity.yaml` | Dynamo aggregated DGD — same engine flags as Standalone, `--router-mode random` (KV routing has no benefit at 1 replica) |
| `run-benchmark-natural-eos.sh` | bench_serving / aiperf with **variable OSL** (natural EOS termination) — matches Google's published methodology, direct apples-to-apples comparison |
| `run-benchmark-parity.sh` | aiperf-based parity benchmark — same workload to both Standalone and Dynamo endpoints, locked OSL=8192, deterministic sampling |
| `benchmark-kimi-k25-nvfp4-pod.yaml` | aiperf client pod (runs the benchmark scripts against either endpoint) |
| [`nvfp4-kv-optimized/`](./nvfp4-kv-optimized/) | **Goal 3 — Dynamo optimized** (KV-aware routing on shared-prefix workload, 2-replica × TP=8 single-node). Validated results: 3.2× sustainable concurrency, 38-57% lower TTFT vs Standalone |

## Topology

```
2× g4-standard-384  (16 GPUs total, RTX PRO 6000 Blackwell SM_120)
├── TP=8 × PP=2     (one distributed SGLang worker across both nodes)
├── DP=8            (DP attention on for MoE)
└── modelopt_fp4    (NVFP4 weights ~370 GB, bf16 KV cache, mem-fraction 0.82)
```

## Recommended: PP=1 (single-node) for optimized latency

For latency-sensitive Dynamo deployments, **prefer PP=1 (single-node TP)** when your hardware allows the model + KV cache to fit on one node — pair it with Dynamo's KV-aware routing (`--router-mode kv`) and multi-replica fan-out.

This recipe uses **PP=2 across 2 nodes** only because Kimi K2.5 NVFP4 (~370 GB) doesn't fit comfortably on a single g4 node with usable KV cache. Cross-node PP is **not the recommended shape for optimized latency** — use it only when single-node fit isn't feasible.

**See [`nvfp4-kv-optimized/`](./nvfp4-kv-optimized/)** (Goal 3) for the PP=1 reference on this hardware: 2 replicas × TP=8 single-node + KV-aware routing on shared-prefix workload. Single-node fit for K2.5 NVFP4 on g4 is memory-tight (`--mem-fraction-static 0.97 --disable-cuda-graph`, ~2.6 GB KV/GPU → concurrency-capped), but it isolates Dynamo's KV-routing benefit cleanly: 3.2× sustainable concurrency and -10% ITL P50 vs single-node Standalone at equal concurrency on shared-prefix workload.

## Quick start

```bash
# Deploy (apply only the variant you want to benchmark — same DGD name swaps with delete+apply)
kubectl apply -f standalone-sglang-kimi-k25-nvfp4.yaml         # bare SGLang (matches Google ref)
kubectl apply -f dgd-agg-sglang-kimi-k25-nvfp4-parity.yaml     # Dynamo parity (random router)
kubectl apply -f benchmark-kimi-k25-nvfp4-pod.yaml             # aiperf client pod

# Wait for engine ready (~12-15 min), then copy scripts to the client pod once:
kubectl cp run-benchmark-natural-eos.sh perf-kimi-k25-nvfp4:/workspace/
kubectl cp run-benchmark-parity.sh      perf-kimi-k25-nvfp4:/workspace/
kubectl exec perf-kimi-k25-nvfp4 -- chmod +x /workspace/run-benchmark-*.sh
```

Pick the right benchmark for what you want to measure:

```bash
# Use case 1 — Goal 1 (NVIDIA Standalone vs Google): variable OSL, bench_serving harness.
# Matches Google's published methodology exactly. Run against the Standalone deployment.
kubectl exec perf-kimi-k25-nvfp4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-natural-eos.sh standalone > /workspace/bench.log 2>&1 &'

# Use case 2 — Goal 2 (Dynamo Parity vs Standalone): locked OSL=8192, aiperf, random workload.
# Run twice — once against Standalone, then tear it down and run against Dynamo parity.
kubectl exec perf-kimi-k25-nvfp4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-parity.sh standalone > /workspace/bench.log 2>&1 &'
kubectl exec perf-kimi-k25-nvfp4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-parity.sh dynamo > /workspace/bench.log 2>&1 &'

# Use case 3 — Goal 3 (Dynamo optimized): KV-aware routing on shared-prefix workload.
# 2-replica × TP=8 single-node, concurrency sweep. See nvfp4-kv-optimized/README.md for setup
# (different topology + DGD from the parity DGD above).
```

Each YAML and script has inline comments explaining the choices.

## Reference

Google's published Kimi K2.5 NVFP4 reference: <https://github.com/shivajid/sglang-rtx-pro-6000/tree/main/models/KimiK2.5/nvfp4>

## Performance Benchmarks

Workload: ISL=1024, OSL=8192, conc=512, 1,536 prompts. **bold** = directly comparable column.

| Variant | SGLang version | Benchmark / OSL | Output Throughput (tok/s) | Total Throughput (tok/s) | ITL P50 |
|---|---|---|---|---|---|
| Google Standalone (published reference) | `lmsysorg/sglang:dev-cu13` | bench_serving, variable OSL | 3,237 | 3,632 | 121 ms |
| **NVIDIA Standalone** (matches Google methodology) | `lmsysorg/sglang:dev-cu13` | bench_serving, variable OSL | **3,374** (+4.2%) | **~3,786** (+4.2%) | **118 ms** (-2.5%) |
| **NVIDIA Standalone (aiperf fixed-OSL baseline)** ← Dynamo apples-to-apples reference | `lmsysorg/sglang:v0.5.10.post1` | **aiperf, locked OSL=8192** | **3,971** | **4,480** | **122.6 ms** |
| Dynamo parity (vs Standalone aiperf fixed-OSL baseline) | `v0.5.10.post1` (bundled in `sglang-runtime:1.1.0`) | aiperf, locked OSL=8192 | 3,723 (-6.2%) | 4,200 (-6.2%) | 128.0 ms (+4.4%) |
| Standalone @ conc=5, 80% shared prefix (Standalone OOM @ conc=6) | `v0.5.10.post1` | aiperf, locked OSL=8192, 80% shared prefix | 132 | — | 33.5 ms |
| Dynamo + KV-aware routing @ conc=5, 80% shared prefix (see [`nvfp4-kv-optimized/`](./nvfp4-kv-optimized/)) | `v0.5.10.post1` (bundled in `sglang-runtime:1.1.0`) | aiperf, locked OSL=8192, 80% shared prefix | 165 (+25%) | — | 30.0 ms (-10%) |

*SGLang version note*: The Standalone fixed-OSL baseline (row 3) is intentionally pinned to `v0.5.10.post1` — the same SGLang version bundled in Dynamo's certified `sglang-runtime:1.1.0` image — so the Dynamo parity comparison (row 4) holds the SGLang code constant and isolates the wrapper effect from upstream SGLang version drift. Rows 1-2 use the rolling `dev-cu13` tag to match Google's published methodology.

**Reading guide:**
- **Goal 1** (NVIDIA Standalone vs Google): direct apples-to-apples — both use `bench_serving` + variable OSL. NVIDIA Standalone matches and slightly exceeds Google's reference on every metric.
- **Goal 2** (Dynamo parity vs NVIDIA Standalone fixed-OSL baseline): completed. Both runs use `aiperf` + locked OSL + the same SGLang version that's bundled in Dynamo's certified runtime image, so the comparison isolates the Dynamo wrapper from engine config differences. Throughput within ~6% of the baseline.
- **Goal 3** (Dynamo optimized): completed in [`nvfp4-kv-optimized/`](./nvfp4-kv-optimized/). KV-aware routing on shared-prefix workload (2-replica × TP=8 single-node). At equal concurrency (conc=5), Dynamo delivers +25% throughput and -10% ITL P50 vs Standalone. Dynamo also sustains 3.2× higher concurrency on the same hardware (conc=16 vs Standalone's conc=5 ceiling).
