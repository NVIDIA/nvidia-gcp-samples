# FlexTensor Core API — Wan2.2-T2V (standalone)

Standalone FlexTensor weight-offload on Wan2.2-T2V-A14B (no serving layer). Each script runs as a self-contained pod; read results via `kubectl logs`.

## Files

| File | Purpose |
|---|---|
| `wan22_bench.py` / `wan22-bench-pod.yaml` | 3-way A/B: no-offload vs FlexTensor vs diffusers-cpu-offload (VRAM saved, overhead %) |
| `wan3_settle.py` / `wan3-settle-pod.yaml` | 3 Wan2.2 models on 1 GPU, pinned, sequential-settle load (density) |

## Results

| Mode | Gen (s) | Overhead | Peak VRAM | VRAM saved |
|---|---:|---:|---:|---:|
| no-offload | 45.0 | — | 73.6 GB | — |
| **FlexTensor (warm)** | **45.3** | **+0.6%** | **21.4 GB** | **71%** |
| diffusers cpu-offload | 69.7 | +55% | 34.9 GB | 53% |

**3 models on 1 GPU** (pinned, sequential settle): 54.9 GB VRAM / ~345 GB host RSS, 45.1 s/gen each. No-offload OOMs at the 2nd model.

## Run

The pod downloads the model (~57 GB) to its HF cache on first run. Then:

```bash
kubectl create configmap wan22-bench --from-file=wan22_bench.py
kubectl apply -f wan22-bench-pod.yaml && kubectl logs -f pod/wan22-bench

kubectl create configmap wan3-settle --from-file=wan3_settle.py
kubectl apply -f wan3-settle-pod.yaml && kubectl logs -f pod/wan3-settle
```

## Key config

Keep the transformer on CPU and move VAE + text-encoder to GPU, then:

```python
OffloadConfig(pinned_memory=True, max_gpu_mem_fraction=0.08,
              include_patterns=["rope","patch_embedding","condition_embedder","blocks.*","norm_out","proj_out"])
```

Lower `max_gpu_mem_fraction` → more offloaded → more models fit **in VRAM** — but once pinned, the actual ceiling is **host RAM** (~83 GB/model here, so ~3 on a 384 GB node). `pinned_memory=True` gives the ~0% warm path (needs the `IPC_LOCK` capability to pin > 8 MB); pageable is ~2× slower but unbounded. Load models one at a time (**sequential settle**) so the un-freed `from_pretrained` footprints don't stack and OOM.

**Alternative — sequential profile:** profile the first model, then reuse that profile on the rest via `flextensor.save_profile(dir, name)` → `flextensor.offload_from_profile(model, dir)`, which skips discovery/profiling on replicas 2…N. VRAM, host-RSS, and steady-state gen latency are identical to sequential-settle — it only trims the one-time profiling pass off replica bring-up — so this sample ships the simpler settle path.
