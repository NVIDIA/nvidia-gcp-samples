# Dynamo on GCP g4 (RTX PRO 6000 Blackwell) — Kimi K2.5 NVFP4

NVIDIA Dynamo + SGLang reference deployment for **Kimi K2.5 NVFP4** on Google Cloud's `g4-standard-384` instance (8× RTX PRO 6000 Blackwell, SM_120). 2-node GKE topology, TP=8 + PP=2.

## What's here

| File | Purpose |
|---|---|
| `standalone-sglang-kimi-k25-nvfp4.yaml` | Bare SGLang StatefulSet (2 nodes, TP=8 + PP=2) — matches Google's published Kimi NVFP4 reference flag-for-flag, but pins SGLang to `v0.5.10.post1` (the version Dynamo's `sglang-runtime:1.1.0` image bundles) for apples-to-apples comparison |
| `dgd-agg-sglang-kimi-k25-nvfp4-parity.yaml` | Dynamo aggregated DGD — same engine flags as Standalone, `--router-mode random` (KV routing has no benefit at 1 replica) |
| `dgd-agg-sglang-kimi-k25-nvfp4-optimized.yaml` | **Template (work in progress)** — Dynamo aggregated DGD with KV-aware routing, radix cache, KV events for shared-prefix workloads. Not yet validated for production; use as a starting point and tune for your workload. |
| `run-benchmark-natural-eos.sh` | bench_serving / aiperf with **variable OSL** (natural EOS termination) — matches Google's published methodology, direct apples-to-apples comparison |
| `run-benchmark-parity.sh` | aiperf-based parity benchmark — same workload to both Standalone and Dynamo endpoints, locked OSL=8192, deterministic sampling |
| `run-benchmark-optimized.sh` | **Template (work in progress)** — sample shared-prefix benchmark (default 80% shared, configurable via `SHARED_PERCENT`) for the optimized Dynamo DGD. Use to explore the latency-vs-throughput trade-off; tune workload to your real distribution. |
| `benchmark-kimi-k25-nvfp4-pod.yaml` | aiperf client pod (runs the benchmark scripts against either endpoint) |

## Topology

```
2× g4-standard-384  (16 GPUs total, RTX PRO 6000 Blackwell SM_120)
├── TP=8 × PP=2     (one distributed SGLang worker across both nodes)
├── DP=8            (DP attention on for MoE)
└── modelopt_fp4    (NVFP4 weights ~370 GB, bf16 KV cache, mem-fraction 0.82)
```

## Quick start

```bash
# Deploy (apply only the variant you want to benchmark — same DGD name swaps with delete+apply)
kubectl apply -f standalone-sglang-kimi-k25-nvfp4.yaml         # bare SGLang (matches Google ref)
kubectl apply -f dgd-agg-sglang-kimi-k25-nvfp4-parity.yaml     # Dynamo parity (random router)
kubectl apply -f dgd-agg-sglang-kimi-k25-nvfp4-optimized.yaml  # Dynamo optimized (KV router + radix)
kubectl apply -f benchmark-kimi-k25-nvfp4-pod.yaml             # aiperf client pod

# Wait for engine ready (~12-15 min), then copy scripts to the client pod once:
kubectl cp run-benchmark-natural-eos.sh        perf-kimi-k25-nvfp4:/workspace/
kubectl cp run-benchmark-parity.sh             perf-kimi-k25-nvfp4:/workspace/
kubectl cp run-benchmark-optimized.sh   perf-kimi-k25-nvfp4:/workspace/
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

# Use case 3 — Goal 3 (Dynamo optimized): locked OSL, 80% shared prefix (default; override
# via SHARED_PERCENT=98|50|0). Runs against the Dynamo optimized DGD — exercises radix cache
# + KV-aware routing.
kubectl exec perf-kimi-k25-nvfp4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-optimized.sh dynamo > /workspace/bench.log 2>&1 &'
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
| Dynamo optimized — work in progress | `v0.5.10.post1` (bundled in `sglang-runtime:1.1.0`) | aiperf, locked OSL=8192, shared-prefix workload | — | — | — |

*SGLang version note*: The Standalone fixed-OSL baseline (row 3) is intentionally pinned to `v0.5.10.post1` — the same SGLang version bundled in Dynamo's certified `sglang-runtime:1.1.0` image — so the Dynamo parity comparison (row 4) holds the SGLang code constant and isolates the wrapper effect from upstream SGLang version drift. Rows 1-2 use the rolling `dev-cu13` tag to match Google's published methodology.

**Reading guide:**
- **Goal 1** (NVIDIA Standalone vs Google): direct apples-to-apples — both use `bench_serving` + variable OSL. NVIDIA Standalone matches and slightly exceeds Google's reference on every metric.
- **Goal 2** (Dynamo parity vs NVIDIA Standalone fixed-OSL baseline): completed. Both runs use `aiperf` + locked OSL + the same SGLang version that's bundled in Dynamo's certified runtime image, so the comparison isolates the Dynamo wrapper from engine config differences. Throughput within ~6% of the baseline.
- **Goal 3** (Dynamo optimized): work in progress. Dynamo's primary value-add is the KV-aware router + radix cache on shared-prefix workloads, which is the next focus.
