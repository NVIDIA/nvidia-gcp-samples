# Dynamo on GCP g4 (RTX PRO 6000 Blackwell) — Kimi K2.5 Native INT4

NVIDIA Dynamo + SGLang reference deployment for **Kimi K2.5 Native INT4** on Google Cloud's `g4-standard-384` instance (8× RTX PRO 6000 Blackwell, SM_120). 2-node GKE topology, TP=8 + PP=2.

## What's here

| File | Purpose |
|---|---|
| `standalone-sglang-kimi-k25-int4.yaml` | Bare SGLang StatefulSet (2 nodes, TP=8 + PP=2) — matches Google's published Kimi K2.5 INT4 reference flag-for-flag, pinned to SGLang `v0.5.10.post1` (the same version Dynamo's `sglang-runtime:1.1.0` image bundles) for apples-to-apples comparison |
| `dgd-agg-sglang-kimi-k25-int4.yaml` | Dynamo aggregated DGD (parity) — same engine flags as Standalone, `--router-mode random` (KV routing has no benefit at 1 replica) |
| `dgd-agg-sglang-kimi-k25-int4-optimized.yaml` | **Template (work in progress)** — Dynamo aggregated DGD with KV-aware routing, radix cache, KV events for shared-prefix workloads. Not yet validated for production; use as a starting point and tune for your workload. |
| `run-benchmark-natural-eos.sh` | bench_serving / aiperf with **variable OSL** (natural EOS termination) — matches Google's published methodology, direct apples-to-apples comparison |
| `run-benchmark-parity.sh` | aiperf-based parity benchmark — locked OSL=8192, random workload, deterministic sampling (NVIDIA baseline + Dynamo parity comparison) |
| `run-benchmark-optimized.sh` | **Template (work in progress)** — sample shared-prefix benchmark (default 80% shared, configurable via `SHARED_PERCENT`) for the optimized Dynamo DGD. Use to explore the latency-vs-throughput trade-off; tune workload to your real distribution. |
| `benchmark-kimi-k25-int4-pod.yaml` | aiperf client pod (runs the benchmark scripts against either endpoint) |

## Topology

```
2× g4-standard-384  (16 GPUs total, RTX PRO 6000 Blackwell SM_120)
├── TP=8 × PP=2     (one distributed SGLang worker across both nodes)
├── No DP attention (Google's INT4 reference does not enable DP)
└── Native INT4     (Compressed Tensors WNA16 Marlin MoE, fp8_e5m2 KV cache, mem-fraction 0.85)
```

## Quick start

```bash
# Deploy (apply only the variant you want to benchmark — same DGD name swaps with delete+apply)
kubectl apply -f standalone-sglang-kimi-k25-int4.yaml          # bare SGLang (matches Google ref)
kubectl apply -f dgd-agg-sglang-kimi-k25-int4.yaml             # Dynamo parity (random router)
kubectl apply -f dgd-agg-sglang-kimi-k25-int4-optimized.yaml   # Dynamo optimized (KV router + radix)
kubectl apply -f benchmark-kimi-k25-int4-pod.yaml              # aiperf client pod

# Wait for engine ready (~20-25 min cold start]), then copy scripts once:
kubectl cp run-benchmark-natural-eos.sh        perf-kimi-k25-int4:/workspace/
kubectl cp run-benchmark-parity.sh             perf-kimi-k25-int4:/workspace/
kubectl cp run-benchmark-optimized.sh   perf-kimi-k25-int4:/workspace/
kubectl exec perf-kimi-k25-int4 -- chmod +x /workspace/run-benchmark-*.sh
```

Pick the right benchmark for what you want to measure:

```bash
# Use case 1 — Goal 1 (NVIDIA Standalone vs Google): variable OSL, bench_serving harness.
# Matches Google's published methodology exactly. Run against the Standalone deployment.
kubectl exec perf-kimi-k25-int4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-natural-eos.sh standalone > /workspace/bench.log 2>&1 &'

# Use case 2 — Goal 2 (Dynamo Parity vs Standalone): locked OSL=8192, aiperf, random workload.
# Run twice — once against Standalone, then tear it down and run against Dynamo parity.
kubectl exec perf-kimi-k25-int4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-parity.sh standalone > /workspace/bench.log 2>&1 &'
kubectl exec perf-kimi-k25-int4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-parity.sh dynamo > /workspace/bench.log 2>&1 &'

# Use case 3 — Goal 3 (Dynamo optimized): locked OSL, 80% shared prefix (default; override
# via SHARED_PERCENT=98|50|0). Runs against the Dynamo optimized DGD — exercises radix cache
# + KV-aware routing.
kubectl exec perf-kimi-k25-int4 -- bash -c \
  'nohup setsid /workspace/run-benchmark-optimized.sh dynamo > /workspace/bench.log 2>&1 &'
```

Each YAML and script has inline comments explaining the choices.

## Reference

Google's published Kimi K2.5 INT4 reference: <https://github.com/shivajid/sglang-rtx-pro-6000/tree/main/models/KimiK2.5>

## Performance Benchmarks

Workload: ISL=1024, OSL=8192, conc=512, 1,536 prompts. **bold** = directly comparable column.

| Variant | SGLang version | Benchmark / OSL | Output Throughput (tok/s) | Total Throughput (tok/s) | ITL P50 |
|---|---|---|---|---|---|
| Google Standalone (published reference) | `lmsysorg/sglang:v0.5.10.post1` | bench_serving, variable OSL | 3,069 | 3,443 | 134 ms |
| **NVIDIA Standalone** (matches Google methodology) | `lmsysorg/sglang:v0.5.10.post1` | bench_serving, variable OSL | **3,525** (+14.8%) | **3,955** (+14.8%) | **132 ms** (-1.4%) |
| **NVIDIA Standalone (aiperf fixed-OSL baseline)** ← Dynamo apples-to-apples reference | `lmsysorg/sglang:v0.5.10.post1` | **aiperf, locked OSL=8192** | **2,377** | **2,677** | **157 ms** |
| Dynamo parity (vs Standalone aiperf fixed-OSL baseline) | `v0.5.10.post1` (bundled in `sglang-runtime:1.1.0`) | aiperf, locked OSL=8192 | 2,536 (+6.7%) | 2,855 (+6.7%) | 144 ms (-7.9%) |
| Dynamo optimized — work in progress | `v0.5.10.post1` (bundled in `sglang-runtime:1.1.0`) | aiperf, locked OSL=8192, shared-prefix workload | — | — | — |

*SGLang version note*: NVIDIA Standalone and Dynamo runs are all pinned to `v0.5.10.post1` — the same SGLang version bundled in Dynamo's certified `sglang-runtime:1.1.0` image — so every comparison holds the SGLang code constant and isolates only the Dynamo wrapper effect.

*OSL methodology note*: Google's published reference uses **variable OSL** (natural EOS termination, average ~4,189 output tokens per request). The fixed-OSL baseline (row 3 above) and Dynamo parity (row 4) use **locked OSL=8,192** — every request generates exactly 8,192 output tokens — which exposes Dynamo wrapper overhead clearly but is not directly comparable to Google's variable-OSL number.

*Methodology note for Dynamo parity (row 4)*: Engine flags are identical to the Standalone baseline (row 3), Dynamo Frontend uses `--router-mode random`, and the worker explicitly sets `--disable-radix-cache` so the wrapper effect is isolated from engine cache state. 

**Reading guide:**
- **Goal 1** (NVIDIA Standalone vs Google): direct apples-to-apples — both use `bench_serving` + variable OSL on the same SGLang version and engine config. NVIDIA Standalone matches and exceeds Google's published numbers on every metric.
- **Goal 2** (Dynamo parity vs NVIDIA Standalone fixed-OSL baseline): completed. Both runs use `aiperf` + locked OSL + the same SGLang version bundled in Dynamo's certified runtime image, on fresh pods, with radix cache disabled — so the comparison isolates the Dynamo wrapper only. Dynamo parity throughput is +6.7% over the Standalone baseline at locked OSL with random workload, with ITL P50 -7.9% lower.
- **Goal 3** (Dynamo optimized): work in progress. Dynamo's primary value-add is the KV-aware router + radix cache on shared-prefix workloads, which is the next focus.
