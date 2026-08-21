#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dynamo custom worker: Wan2.2-T2V-A14B (Diffusers) + FlexTensor weight-offload, serving /v1/videos.

Forked from ai-dynamo/dynamo `examples/diffusers/worker.py`: the Dynamo wiring (VideoCreateRequest/
Response, @dynamo_endpoint, register_llm(ModelType.Videos), serve_endpoint) is kept; the model load +
generate are swapped for `WanPipeline` + `flextensor.offload()`. `--budget` = max_gpu_mem_fraction;
`--unpinned` = pageable host memory (~2x). Passing >1 name to `--models` serves that many models on
one GPU (sequential-settle) for the density demo.
"""
import argparse
import asyncio
import base64
import logging
import os
import resource
import tempfile
import time
import uuid

import torch
import uvloop
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from pydantic import BaseModel, Field

import flextensor
from flextensor import OffloadConfig
from dynamo.llm import ModelInput, ModelType, WorkerType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DT = torch.bfloat16
# Official FlexTensor Wan include_patterns (flextensor examples/diffusers/quickstart/wan_t2v.py).
WAN_INCLUDE = ["rope", "patch_embedding", "condition_embedder", "blocks.*", "norm_out", "proj_out"]


def _ns() -> str:
    ns = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    return f"{ns}-{suffix}" if suffix else ns


# ── Request / Response (matches dynamo examples/diffusers/worker.py; Wan defaults) ──
class NvExt(BaseModel):
    fps: int = 16
    num_frames: int | None = 17
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    seed: int | None = 0
    negative_prompt: str | None = None


class VideoCreateRequest(BaseModel):
    prompt: str
    model: str
    size: str = "832x480"
    seconds: int = 1
    user: str | None = None
    nvext: NvExt = Field(default_factory=NvExt)


class VideoData(BaseModel):
    # Matches dynamo 1.3.0 NvVideosResponse.VideoData — output_format is REQUIRED by the frontend.
    output_format: str = "mp4"
    url: str | None = None
    b64_json: str | None = None


class VideoCreateResponse(BaseModel):
    id: str
    object: str = "video"
    model: str
    status: str = "completed"
    progress: int = 100
    created: int
    data: list[VideoData] = []
    error: str | None = None
    inference_time_s: float | None = None


# ── Backend: one offloaded Wan pipeline per served model name ──
class WanFlexTensorBackend:
    def __init__(self, served_name: str, model_path: str, budget: float, pinned: bool, tag: str) -> None:
        self.served_name = served_name  # /v1/videos routing key (distinct per replica)
        self.model_path = model_path    # HF weights (from_pretrained); shared across replicas
        self.budget = budget
        self.pinned = pinned
        self.tag = tag  # unique flextensor manager-name prefix (for multi-model density)
        self._lock = asyncio.Lock()
        self.pipe = None

    def _load(self):
        pipe = WanPipeline.from_pretrained(self.model_path, torch_dtype=DT)
        # Keep transformer(s) on CPU for offload; move the rest to GPU.
        for name, comp in pipe.components.items():
            if isinstance(comp, torch.nn.Module) and name not in ("transformer", "transformer_2"):
                comp.to("cuda")
        cfg = OffloadConfig(
            enabled=True, gpu_device=0, discovery_iters=2, profiling_iters=3, min_blocks=2,
            max_gpu_mem_fraction=self.budget, pinned_memory=self.pinned, include_patterns=WAN_INCLUDE,
        )
        pipe.transformer = flextensor.offload(pipe.transformer, config=cfg, name=f"{self.tag}_transformer")
        pipe.transformer_2 = flextensor.offload(pipe.transformer_2, config=cfg, name=f"{self.tag}_transformer2")
        # Settle: one gen drives DISCOVERY->PROFILING->INFERENCE (frees source, packs pinned) so the
        # first real request is warm and (for density) the from_pretrained transient is released now.
        pipe(prompt="warmup", height=480, width=832, num_frames=17, num_inference_steps=20, guidance_scale=4.0)
        return pipe

    async def initialize_model(self) -> None:
        logger.info("Loading Wan+FlexTensor served=%s weights=%s budget=%.2f pinned=%s",
                    self.served_name, self.model_path, self.budget, self.pinned)
        loop = asyncio.get_running_loop()
        self.pipe = await loop.run_in_executor(None, self._load)
        logger.info("READY served=%s | GPU resident %.1f GB | peak %.1f GB | host RSS %.0f GB",
                    self.served_name, torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)

    def _gen_mp4(self, prompt, w, h, num_frames, steps, gs, seed, neg) -> bytes:
        gen = None if seed is None else torch.Generator(device="cuda").manual_seed(int(seed))
        out = self.pipe(prompt=prompt, negative_prompt=neg, height=h, width=w, num_frames=num_frames,
                        num_inference_steps=steps, guidance_scale=gs, generator=gen)
        frames = out.frames[0]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.mp4")
            export_to_video(frames, p, fps=16)
            with open(p, "rb") as f:
                return f.read()

    @dynamo_endpoint(VideoCreateRequest, VideoCreateResponse)
    async def create_video(self, request: VideoCreateRequest):
        if self.pipe is None:
            raise RuntimeError("pipeline not initialized")
        w, h = (int(x) for x in request.size.lower().split("x", 1))
        nv = request.nvext
        nf = nv.num_frames if nv.num_frames is not None else nv.fps * request.seconds
        vid = f"video_{uuid.uuid4().hex}"
        ts = int(time.time())
        async with self._lock:  # WanPipeline not re-entrant
            t = time.perf_counter()
            logger.info("[%s] gen model=%s %dx%d frames=%d steps=%d", vid, request.model, w, h, nf, nv.num_inference_steps)
            mp4 = await asyncio.to_thread(self._gen_mp4, request.prompt, w, h, nf,
                                          nv.num_inference_steps, nv.guidance_scale, nv.seed, nv.negative_prompt)
            elapsed = time.perf_counter() - t
            logger.info("[%s] done %.1fs | %.2f MB", vid, elapsed, len(mp4) / 1_048_576)
            yield VideoCreateResponse(
                id=vid, created=ts, model=request.model, inference_time_s=elapsed,
                data=[VideoData(output_format="mp4", b64_json=base64.b64encode(mp4).decode())],
            ).model_dump()


async def _register(endpoint, served_name: str, model_path: str) -> None:
    await register_llm(ModelInput.Text, ModelType.Videos, endpoint, model_path, served_name,
                       worker_type=WorkerType.Aggregated)
    logger.info("registered /v1/videos served=%s weights=%s", served_name, model_path)


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    served = [m.strip() for m in args.models.split(",") if m.strip()]
    ns = _ns()
    coros = []
    for i, served_name in enumerate(served):
        # Each served name = its own endpoint + offloaded pipeline (sequential settle inside initialize_model).
        ep = runtime.endpoint(f"{ns}.backend{i}.generate")
        backend = WanFlexTensorBackend(served_name, args.model_path, args.budget, not args.unpinned, tag=f"m{i}")
        await backend.initialize_model()  # sequential: settle replica i before loading i+1
        logger.info("serving %s (weights=%s) on %s.backend%d.generate", served_name, args.model_path, ns, i)
        coros.append(ep.serve_endpoint(backend.create_video))
        coros.append(_register(ep, served_name, args.model_path))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.info("ALL %d REPLICAS LOADED | peak host RSS %.0f GB | GPU peak %.1f GB",
                len(served), peak_rss, torch.cuda.max_memory_allocated() / 1e9)
    await asyncio.gather(*coros)


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    db = os.environ.get("DYN_DISCOVERY_BACKEND") or ("kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file")
    logger.info("discovery=%s namespace=%s request-plane=tcp", db, _ns())
    runtime = DistributedRuntime(loop, db, "tcp")
    await backend_worker(runtime, args)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan2.2 + FlexTensor Dynamo /v1/videos worker")
    p.add_argument("--models", default=DEFAULT_MODEL,
                   help="comma-separated served model names (>1 = density on one GPU)")
    p.add_argument("--model-path", default=DEFAULT_MODEL, dest="model_path",
                   help="HF weights path (from_pretrained); shared by all served names")
    p.add_argument("--budget", type=float, default=0.30, help="max_gpu_mem_fraction (0.30 single, 0.08 for 3-on-1)")
    p.add_argument("--unpinned", action="store_true", help="pageable host mem (~2x) instead of pinned")
    return p.parse_args()


if __name__ == "__main__":
    _a = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    uvloop.install()
    asyncio.run(main(_a))
