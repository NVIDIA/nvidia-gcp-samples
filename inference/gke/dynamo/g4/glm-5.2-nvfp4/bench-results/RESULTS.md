# GLM-5.2-NVFP4 on RTX PRO 6000 (SM120) — Functional Benchmark

- **Workload:** `sglang.bench_serving`, random dataset, 16 prompts, ISL 128 / OSL 64, request rate 4.
- **Setup:** TP=8 on one GKE `g4-standard` node (8× RTX PRO 6000, PCIe), recipe per `../README.md`,
  default config (prefix/radix caching on).
- **Method:** standalone benched via the native `--backend sglang` endpoint (:30000); Dynamo via
  `--backend sglang-oai-chat` through the frontend (:8000). Reported runs are **warm** (JIT/autotune
  caches populated, second bench of the deployment). Cold first launches spend their first bench on
  one-time JIT compilation (~29 tok/s, ~6 s TTFT) and are not representative. The bench uses a fixed
  seed, so the warm run's prefills partially hit the prefix cache — identically for both columns.

## Results (warm)

| Metric | SGLang standalone | Dynamo (aggregated) | Δ |
|---|---|---|---|
| Successful requests | 16/16 | 16/16 | — |
| Output tok/s | 121.08 | 127.08 | +5.0% |
| Median TTFT (ms) | 127.0 | 128.6 | +1.3% |
| Median TPOT (ms) | 47.1 | 42.4 | −9.8% |
| Median ITL (ms) | 33.5 | 31.7 | −5.2% |

Functional validation numbers, not performance-tuned. The two columns use different bench protocols
(native completion vs OpenAI-chat), so small deltas reflect protocol/token accounting — the takeaway
is that Dynamo's routing layer adds no measurable cost on a single aggregated node. A cache-disabled
A/B (`--disable-radix-cache`) showed unchanged correctness and per-token decode latency (Dynamo TPOT
identical), so prefix caching on is the recommended configuration; disable it only for clean-prefill
benchmarking.
