# FlexTensor Weight-Offload on g4 (RTX PRO 6000 / SM120)

[NVIDIA FlexTensor](https://github.com/ai-dynamo/flextensor) offloads model **weights** from GPU VRAM to host RAM and streams each layer back over PCIe just-in-time (prefetched behind compute), so you can **fit a bigger model** — or **more models per GPU** — at near-zero latency cost when the workload is compute-bound. This sample demonstrates it on **Wan2.2-T2V-A14B** (text-to-video diffusion) on a GKE `g4-standard` node, **standalone** and served through **[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)**.

Diffusion denoising is a big, compute-bound per-step GEMM, so FlexTensor prefetches each transformer layer to the GPU while the previous one computes — the PCIe transfer hides behind compute and the offload is nearly free on the warm path. The bottleneck moves from **VRAM (96 GB)** to **host RAM (384 GB)**.

## Contents

| Folder | What |
|---|---|
| [`core-api/`](core-api/) | Standalone (FlexTensor core API): offload-benefit A/B + 3 models on 1 GPU (density) |
| [`dynamo/`](dynamo/) | Same offload served through a Dynamo custom `/v1/videos` worker + benchmark |

## Results (measured on g4, Wan2.2-T2V-A14B)

| | Result |
|---|---|
| **Offload benefit** (1 model) | **71% VRAM saved** (73.6 → 21.4 GB) at **+0.6%** warm overhead |
| **Density** (3 models, 1 GPU, pinned) | 3× co-resident, 54.9 GB VRAM / ~345 GB host RSS, 45.1 s/gen each (no-offload OOMs at 2) |
| **Served via Dynamo** | `POST /v1/videos` → MP4, VRAM 21.4 GB, **< 1% serving overhead**, 3 models behind one endpoint |

Density is a **capacity** win (host many models per GPU, routed on demand), not a throughput multiplier — one GPU's compute is shared across the resident models.

## When to use

Offload is near-free only when each step is **compute-bound** enough to hide the PCIe weight transfer:
- ✅ **Weight-heavy, compute-bound** — image/video diffusion (validated here on Wan2.2). Large-LLM **prefill** is the other classic compute-bound case where the same principle applies.
- ❌ **Weight-light or memory-bound** — small TTS/ASR, LLM **decode**: little to offload and the transfer can't hide → overhead, not gain. Use batching / KV management instead.

After offloading, size against **host RAM** (pinned footprint), not VRAM.

## Next steps

FlexTensor **0.4.0** (2026-08-18) added **vLLM support** plus performance fixes; serving quantized LLMs with prefill weight-offload via the Dynamo vLLM worker is a pending follow-up. This sample covers the validated **diffusion** path.

## References

- FlexTensor: <https://github.com/ai-dynamo/flextensor>
- NVIDIA Dynamo: <https://github.com/ai-dynamo/dynamo> (`examples/diffusers/worker.py`)
- Model: <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>
