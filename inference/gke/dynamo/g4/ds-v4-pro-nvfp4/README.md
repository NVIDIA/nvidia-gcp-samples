# DeepSeek-V4-Pro-NVFP4 on RTX Pro 6000 Guide

Serve `nvidia/DeepSeek-V4-Pro-NVFP4` (910B, mixed FP8 + NVFP4 experts) on **RTX Pro 6000 (SM120, PCIe-only) GPUs on GKE**. Two deployment paths, both **TP=8 PP=2 across 2 nodes**:
- **A) Standalone SGLang** (no Dynamo) — `standalone-sglang-dsv4-pro-nvfp4-mn.yaml`
- **B) Dynamo DGD** (community SGLang image + Dynamo wheel) — `dgd-sglang-dsv4-pro-nvfp4-v0514.yaml`

**Platform:** Google Kubernetes Engine (GKE), `g4-standard-384` nodes (8× RTX Pro 6000 Blackwell per node). This guide assumes the GKE cluster and the Dynamo platform (operator + etcd + NATS) are already provisioned — for cluster and platform setup, refer to the official Dynamo GKE guide: https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/cloud-providers/gke/gke.md

**Engine image (both paths):** `lmsysorg/sglang:v0.5.14-cu129` — SGLang 0.5.14, which includes [PR #25820](https://github.com/sgl-project/sglang/pull/25820) (DeepSeek-V4 NVFP4 MoE support) plus flashinfer 0.6.12. Path A runs this image directly; path B runs the same image and installs the Dynamo wheel with `--no-deps`, which preserves this exact SGLang build — so PR #25820 is present in both runs.

**Required for SM120:** `--moe-runner-backend flashinfer_cutlass`. The default `flashinfer_trtllm_routed` MoE kernel is **SM100-only** (`_sm100f`, the datacenter Blackwell architecture) and fails at the GEMM on SM120; `flashinfer_cutlass` provides the SM120 kernels (`flashinfer/fused_moe/cute_dsl/blackwell_sm12x`). **CPU offload is not used** (not supported for these NVFP4 weights — an SGLang offloader limitation); memory fit is achieved with `--mem-fraction-static 0.62` (a compact KV pool).

## Results — sample workload

Workload: random, 16 prompts, ISL 128 / OSL 64, rate 4 req/s.

| Metric | Standalone | Dynamo | Dynamo (warmed up) |
|---|---|---|---|
| Output tok/s | 34.79 | 33.10 | 40 – 43 |
| Median TTFT | 12.8 s | 9.5 s | 6.1 – 7.5 s |
| Median TPOT / ITL | 144 / 129 ms | 144 / 131 ms | 150 / 128 ms |

**Decode latency (TPOT/ITL) is identical** across all (same engine + SM120 CUTLASS kernels — Dynamo adds no per-token cost). Dynamo starts within ~5% of standalone throughput and, **once warmed up (radix + autotune cache), reaches 40–43 tok/s with ~6 s TTFT** — Dynamo's caching/orchestration delivering higher throughput and lower first-token latency. The absolute numbers are modest by design (PP=2 over PCIe, no InfiniBand, CUDA graphs disabled) — a functional validation, not a performance-tuned result. Full detail: `bench-results/RESULTS.md`.

## Prerequisites

- `kubectl` context set to the target GKE cluster (RTX Pro 6000 node pool).
- **2 RTX Pro 6000 nodes** (each path uses TP=8 PP=2 = 2 nodes); run **one** path at a time.
- The model is already staged in the `model-cache` PVC at `…/models--nvidia--DeepSeek-V4-Pro-NVFP4/snapshots/latest`.
- Path B only: the Dynamo platform (operator + etcd + NATS) must be installed in the cluster (already present here).

---

## A) Standalone SGLang (no Dynamo)

```bash
cd dynamo-g4/ds-v4-pro-nvfp4

# 1. Deploy (StatefulSet: leader + worker on 2 nodes)
kubectl apply -f standalone-sglang-dsv4-pro-nvfp4-mn.yaml

# 2. Wait for ready (~13 min: cross-node load + FP4 autotune)
kubectl logs -f sglang-dsv4-mn-0 | grep -m1 -E "fired up and ready to roll"

# 3. Smoke test (the engine serves on the leader pod, port 30000)
kubectl exec sglang-dsv4-mn-0 -- curl -s --max-time 120 http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/DeepSeek-V4-Pro-NVFP4","messages":[{"role":"user","content":"What is tensor parallelism?"}],"max_tokens":64,"temperature":0}'

# 4. Benchmark — native /generate, run inside the leader pod (sglang.bench_serving is built in)
MODEL=/model-cache/.model-express/cache/models--nvidia--DeepSeek-V4-Pro-NVFP4/snapshots/latest
kubectl exec sglang-dsv4-mn-0 -- bash -c "mkdir -p /tmp/b; nohup python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model nvidia/DeepSeek-V4-Pro-NVFP4 --tokenizer $MODEL \
  --dataset-name random --num-prompts 16 --random-input-len 128 --random-output-len 64 \
  --random-range-ratio 1.0 --request-rate 4 --output-file /tmp/b/out.jsonl > /tmp/b/bench.log 2>&1 &"

# 5. Read the result
kubectl exec sglang-dsv4-mn-0 -- sed -n '/Serving Benchmark Result/,/P99 ITL/p' /tmp/b/bench.log

# 6. Tear down (frees the 2 nodes for path B)
kubectl delete -f standalone-sglang-dsv4-pro-nvfp4-mn.yaml
```

---

## B) Dynamo DGD (community image + Dynamo wheel)

The DGD worker and frontend run the **community SGLang image** and install the Dynamo wheel at startup with `pip install --no-deps ai-dynamo …` (which preserves the image's SGLang 0.5.14), then launch `dynamo.sglang` / `dynamo.frontend`. This is handled within the manifest; apply it directly.

```bash
cd dynamo-g4/ds-v4-pro-nvfp4

# 1. Deploy (the operator schedules a Grove gang: agg-ldr + agg-wkr on 2 GPU nodes, frontend on a CPU node)
kubectl apply -f dgd-sglang-dsv4-pro-nvfp4-v0514.yaml

# 2. Wait for READY=True (~15 min: wheel install + cross-node load + FP4 autotune)
kubectl get dynamographdeployment dsv4-sglang-v0514 -w     # Ctrl-C once READY=True

# 3. Smoke test — through the Dynamo frontend (OpenAI API on :8000)
FE=$(kubectl get pods --no-headers | grep 'dsv4-sglang-v0514.*frontend' | awk '{print $1}')
kubectl exec "$FE" -c main -- curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/DeepSeek-V4-Pro-NVFP4","messages":[{"role":"user","content":"Name the Red Planet in one word."}],"max_tokens":16,"temperature":0}'

# 4. Benchmark — via the frontend (OpenAI chat), run inside the frontend pod
MODEL=/model-cache/.model-express/cache/models--nvidia--DeepSeek-V4-Pro-NVFP4/snapshots/latest
kubectl exec "$FE" -c main -- bash -c "mkdir -p /tmp/b; nohup python3 -m sglang.bench_serving \
  --backend sglang-oai-chat --base-url http://127.0.0.1:8000 \
  --model nvidia/DeepSeek-V4-Pro-NVFP4 --tokenizer $MODEL \
  --dataset-name random --num-prompts 16 --random-input-len 128 --random-output-len 64 \
  --random-range-ratio 1.0 --request-rate 4 --output-file /tmp/b/out.jsonl > /tmp/b/bench.log 2>&1 &"

# 5. Read the result
kubectl exec "$FE" -c main -- sed -n '/Serving Benchmark Result/,/P99 ITL/p' /tmp/b/bench.log

# 6. Tear down
kubectl delete -f dgd-sglang-dsv4-pro-nvfp4-v0514.yaml
```

---

## Keynotes

- **`flashinfer_cutlass` is required on SM120.** `flashinfer_trtllm_routed` loads and autotunes, then fails at the MoE GEMM (`trtllm_batched_gemm_runner.cu:286`, `_sm100f` kernel — no SM120 build). RTX Pro 6000 (SM120) has FP4 tensor cores, so this is a missing kernel build, not a hardware limitation.
- **`--cpu-offload-gb` is not supported** for these NVFP4 weights: the SGLang offloader fails on the swizzled block-scale tensors (`offloader.py`, `functional_call` tied-tensor error). Fit memory with 2 nodes and a compact KV pool instead.
- **The DGD frontend requires a memory request** (set in the manifest: `requests.memory 4Gi`). Without it, the best-effort frontend can be evicted under node memory pressure, which restarts the worker gang.
- **Run benchmarks detached** (`nohup … &`, as shown) so that a `kubectl exec` disconnect does not terminate the run.
- **The two paths use different client backends by design**: standalone uses the native `/generate` API (`--backend sglang`); the Dynamo frontend exposes the OpenAI API (`--backend sglang-oai-chat`). Keep the *workload parameters* identical for a fair comparison.

## Files

| File | Purpose |
|---|---|
| `standalone-sglang-dsv4-pro-nvfp4-mn.yaml` | Path A — standalone SGLang, TP=8 PP=2, 2-node StatefulSet + headless Service |
| `dgd-sglang-dsv4-pro-nvfp4-v0514.yaml` | Path B — Dynamo DGD (community image + Dynamo wheel, `--no-deps`) |
| `bench-results/RESULTS.md` | Full results, standalone-vs-Dynamo comparison, and SM120 evidence |
| `failure-analysis.md` | Investigation log: configurations tested, root causes, and the SM120 kernel finding |
| `archive/` | Superseded test manifests (dev.3 w13, vLLM, single-node OOM, CPU-offload), retained for reference |
