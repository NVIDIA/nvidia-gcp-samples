# GLM-5.2-NVFP4 on RTX PRO 6000 (SM120) — Functional Benchmark

- **Workload:** `sglang.bench_serving`, random dataset, 16 prompts, ISL 128 / OSL 64, request rate 4.
- **Setup:** TP=8 on one GKE `g4-standard` node (8× RTX PRO 6000, PCIe), recipe per `../README.md`.
- **Method:** standalone benched via the native `--backend sglang` endpoint (:30000); Dynamo via
  `--backend sglang-oai-chat` through the frontend (:8000). Reported runs are **warm** (JIT/autotune
  caches populated, second bench of the deployment). Cold first launches spend their first bench on
  one-time JIT compilation (~29 tok/s, ~6 s TTFT) and are not representative.

## Results (warm)

| Metric | SGLang standalone | Dynamo (aggregated) | Δ |
|---|---|---|---|
| Successful requests | 16/16 | 16/16 | — |
| Output tok/s | 112.57 | 124.45 | +10.6% |
| Total tok/s | 401.72 | 444.11 | — |
| Median TTFT (ms) | 240.9 | 139.1 | −42.3% |
| Median TPOT (ms) | 54.0 | 42.4 | −21.5% |
| Median ITL (ms) | 38.0 | 32.3 | −15.0% |

Functional validation numbers, not performance-tuned. The two columns use different bench protocols
(native completion vs OpenAI-chat), so small deltas reflect protocol/token accounting — the takeaway
is that Dynamo's routing layer adds no measurable cost on a single aggregated node.
