# FlexTensor via NVIDIA Dynamo — Wan2.2-T2V

The same FlexTensor offload, served behind a Dynamo custom worker on the `/v1/videos` endpoint. `/v1/videos` is capability-routed (any worker that registers `ModelType.Videos` owns it), so no SGLang/vLLM is involved.

## Files

| File | Purpose |
|---|---|
| `wan_ft_dynamo_worker.py` | Dynamo `/v1/videos` worker (Wan2.2 + `flextensor.offload()`) |
| `wan-ft-dynamo-pod.yaml` | Serve 1 model (frontend + worker) — smoke test |
| `wan-ft-dynamo3-pod.yaml` | 3 models on 1 GPU (density) + in-pod benchmark |
| `bench_videos.py` | Async `/v1/videos` benchmark client |

## Results

Measured on g4 (Wan2.2-T2V-A14B) — the Dynamo serving layer adds negligible overhead vs standalone:

| Config | Warm gen/model | Peak VRAM | Serving overhead |
|---|---:|---:|---:|
| **1 model** | **45.1 s** | **21.4 GB** | **< 1%** vs standalone |
| **3 models, 1 GPU** | 45.1 s | 54.9 GB | **< 1%** |

`POST /v1/videos` returns a base64 MP4. The 3 models (`wan-a/b/c`) share one GPU (~345 GB host RSS) behind one endpoint, each routable by name — density is a **capacity** win (more models per GPU), not a throughput multiplier (one GPU's compute is shared).

## Run

Provide a PVC named `model-cache` for the HF cache (or edit the pods to use an `emptyDir`). Single node, file discovery + TCP request plane — no etcd/NATS.

```bash
kubectl create configmap wan-ft-dynamo --from-file=wan_ft_dynamo_worker.py
kubectl apply -f wan-ft-dynamo-pod.yaml && kubectl logs -f pod/wan-ft-dynamo
# → POST /v1/videos returns a base64 MP4

kubectl create configmap wan-ft-dynamo3 \
  --from-file=wan_ft_dynamo_worker.py --from-file=bench_videos.py
kubectl apply -f wan-ft-dynamo3-pod.yaml && kubectl logs -f pod/wan-ft-dynamo3
```

## Notes

- The worker forks ai-dynamo/dynamo `examples/diffusers/worker.py` and swaps FastVideo for `WanPipeline` + `flextensor.offload()`.
- `DYN_DISCOVERY_BACKEND=file` + `DYN_REQUEST_PLANE=tcp` (single-node; tcp is required — video base64 exceeds the NATS 1 MB limit).
- The `/v1/videos` response must match the installed Dynamo version's schema (`output_format` field in Dynamo 1.3.0).
