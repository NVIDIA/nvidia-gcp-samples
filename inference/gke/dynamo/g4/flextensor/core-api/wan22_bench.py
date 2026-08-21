#!/usr/bin/env python3
"""Wan2.2-T2V-A14B on g4 (RTX PRO 6000 / SM120): 3-way A/B to prove FlexTensor's
offload benefit — no-offload vs FlexTensor vs diffusers-cpu-offload.
Measures gen time (overhead %) and peak VRAM (% saved)."""
import time, traceback
import torch, torch.nn as nn

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DT = torch.bfloat16
H, W, FRAMES, STEPS, GS = 480, 832, 17, 20, 4.0
PROMPT = "A cat walking on grass in warm sunlight, cinematic, highly detailed"

from diffusers import WanPipeline
import flextensor
from flextensor import OffloadConfig
print("versions:", "torch", torch.__version__, "| flextensor", flextensor.__version__,
      "| GPU", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

def load():
    return WanPipeline.from_pretrained(MODEL, torch_dtype=DT)

def gen(pipe, tag):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.time()
    pipe(prompt=PROMPT, height=H, width=W, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=GS)
    torch.cuda.synchronize(); dt = time.time() - t
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{tag}] gen {dt:.1f}s | peak_vram {vram:.2f} GB", flush=True)
    return dt, vram

res = {}

# 1) no-offload baseline
try:
    pipe = load().to("cuda")
    res["no-offload"] = gen(pipe, "no-offload")
    del pipe; torch.cuda.empty_cache()
except Exception:
    traceback.print_exc(); res["no-offload"] = (float("nan"), float("nan"))

# 2) FlexTensor offload of both experts (CPU-start + Wan-style patterns + forced budget)
try:
    pipe = load()
    pipe.vae.to("cuda")
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cuda")
    for attr in ["transformer", "transformer_2"]:
        m = getattr(pipe, attr, None)
        if m is None:
            continue
        pats = ["%s.*" % n if isinstance(c, nn.ModuleList) else n for n, c in m.named_children()]
        gb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9
        print(f"offloading {attr}: {gb:.1f} GB, patterns={pats}", flush=True)
        cfg = OffloadConfig(enabled=True, gpu_device=0, discovery_iters=3, profiling_iters=10,
                            min_blocks=2, max_gpu_mem_fraction=0.30, include_patterns=pats)
        setattr(pipe, attr, flextensor.offload(m, config=cfg, name=attr))
    gen(pipe, "flextensor-cold (incl profiling)")   # first run does discovery+profiling
    res["flextensor"] = gen(pipe, "flextensor-warm")  # steady-state = the fair number
    del pipe; torch.cuda.empty_cache()
except Exception:
    traceback.print_exc(); res["flextensor"] = (float("nan"), float("nan"))

# 3) diffusers native cpu-offload
try:
    pipe = load()
    pipe.enable_model_cpu_offload()
    res["diffusers-cpu-offload"] = gen(pipe, "diffusers-cpu-offload")
    del pipe; torch.cuda.empty_cache()
except Exception:
    traceback.print_exc(); res["diffusers-cpu-offload"] = (float("nan"), float("nan"))

# table
t0, m0 = res.get("no-offload", (float("nan"), float("nan")))
print("\n================ RESULT: Wan2.2-T2V-A14B on g4 (RTX PRO 6000 / SM120) ================", flush=True)
print(f"{'Mode':<24}{'Gen(s)':>10}{'Overhead':>12}{'PeakVRAM(GB)':>14}{'MemSaved':>10}", flush=True)
for mode in ["no-offload", "flextensor", "diffusers-cpu-offload"]:
    t, m = res.get(mode, (float("nan"), float("nan")))
    ov = "—" if mode == "no-offload" else f"{(t/t0-1)*100:+.1f}%"
    sv = "—" if mode == "no-offload" else f"{(1-m/m0)*100:.1f}%"
    print(f"{mode:<24}{t:>10.1f}{ov:>12}{m:>14.2f}{sv:>10}", flush=True)
print("=====================================================================================", flush=True)
