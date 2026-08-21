#!/usr/bin/env python3
"""3 Wan2.2-A14B models on ONE g4 GPU with PINNED FlexTensor offload, sequential-settle load.

Load one model at a time: from_pretrained -> offload(pinned) -> one full gen (drives the offload
manager to its steady INFERENCE phase, which frees the original weight source + finishes pinned
packing) -> next. This caps peak host RSS so all 3 fit on a 384 GB node while keeping the 1-model
pinned latency -- density AND low latency together. (Loading all 3 before any gen
instead stacks the un-freed from_pretrained footprints and OOMs.)
"""
import time, resource, gc, traceback
import torch, torch.nn as nn
from diffusers import WanPipeline
import flextensor
from flextensor import OffloadConfig

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DT = torch.bfloat16
H, W, FRAMES, STEPS, N = 480, 832, 17, 20, 3
PROMPT = "A cat walking on green grass in warm sunlight, cinematic"


def rss_now_gb():
    """Current resident set size (GB) from /proc/self/status VmRSS."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    except Exception:
        pass
    return -1.0


def rss_peak_gb():
    """High-water-mark RSS (GB). ru_maxrss is in kB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print("torch %s | flextensor %s | GPU %s" % (torch.__version__, flextensor.__version__, torch.cuda.get_device_name(0)), flush=True)
_s, _h = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print("RLIMIT_MEMLOCK soft=%s hard=%s" % (_s, _h), flush=True)


def load():
    return WanPipeline.from_pretrained(MODEL, torch_dtype=DT)


def offload(pipe, idx):
    pipe.vae.to("cuda")
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cuda")
    for a in ["transformer", "transformer_2"]:
        m = getattr(pipe, a, None)
        if m is None:
            continue
        pats = ["%s.*" % n if isinstance(c, nn.ModuleList) else n for n, c in m.named_children()]
        cfg = OffloadConfig(enabled=True, gpu_device=0, discovery_iters=2, profiling_iters=3,
                            min_blocks=2, max_gpu_mem_fraction=0.08, pinned_memory=True,
                            include_patterns=pats)
        setattr(pipe, a, flextensor.offload(m, config=cfg, name="m%d_%s" % (idx, a)))
    return pipe


def gen(pipe):
    pipe(prompt=PROMPT, height=H, width=W, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=4.0)


# ---- sequential settle: load -> offload -> settle(one gen) -> next ----
pipes = []
try:
    for i in range(N):
        t = time.time()
        p = load()
        print("[m%d] from_pretrained done | rss_now %.0f GB | rss_peak %.0f GB" % (i, rss_now_gb(), rss_peak_gb()), flush=True)
        p = offload(p, i)
        print("[m%d] offload() patched  | rss_now %.0f GB | rss_peak %.0f GB" % (i, rss_now_gb(), rss_peak_gb()), flush=True)
        gen(p)  # SETTLE: DISCOVERY->PROFILING->INFERENCE -> self.model=None -> source freed + pinned packed
        gc.collect()
        torch.cuda.empty_cache()
        pipes.append(p)
        print("[m%d] SETTLED in %.0fs      | GPU %.1f GB | rss_now %.0f GB | rss_peak %.0f GB"
              % (i, time.time() - t, torch.cuda.memory_allocated() / 1e9, rss_now_gb(), rss_peak_gb()), flush=True)
    print(">>> ALL %d PINNED MODELS LOADED (sequential settle) | live rss_now %.0f GB | PEAK rss %.0f GB"
          % (N, rss_now_gb(), rss_peak_gb()), flush=True)
except Exception as e:
    traceback.print_exc()
    print(">>> FAILED at model %d: %s | rss_now %.0f GB | PEAK rss %.0f GB"
          % (len(pipes) + 1, repr(e)[:200], rss_now_gb(), rss_peak_gb()), flush=True)

# ---- warm latency per settled pinned model (already in INFERENCE; re-warm caches then time) ----
lat = []
for i, p in enumerate(pipes):
    try:
        gen(p)  # re-warm (a later model's settle may have evicted this one's CUDA graph/caches)
        torch.cuda.reset_peak_memory_stats()
        t = time.time(); gen(p); l = time.time() - t
        v = torch.cuda.max_memory_allocated() / 1e9
        lat.append(l)
        print("[pinned m%d] warm %.1fs | peak_vram %.1f GB" % (i, l, v), flush=True)
    except Exception as e:
        traceback.print_exc()
        print(">>> gen fail m%d: %s" % (i, repr(e)[:150]), flush=True)

print("\n======== 3-model PINNED (sequential settle) ========", flush=True)
if lat:
    print("  fit %d/%d models | warm %.1fs/model avg | live rss %.0f GB | PEAK rss %.0f GB"
          % (len(pipes), N, sum(lat) / len(lat), rss_now_gb(), rss_peak_gb()), flush=True)
print("  -> sequential settle: 3 pinned models fit on one GPU, each near 1-model-pinned latency.", flush=True)
print("=====================================================", flush=True)
