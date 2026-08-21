#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Async benchmark for the Dynamo /v1/videos endpoint (Wan2.2 + FlexTensor worker).

Fires N requests at a concurrency level, round-robining across the given served model names,
and reports per-request gen latency (p50/p99), throughput (req/s, videos/min), and errors.
Used to quantify (a) Dynamo serving overhead vs standalone and (b) 3-on-1 density
through the router. aiperf doesn't cover /v1/videos, so this is a purpose-built client.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp


async def one_request(session, url, model, prompt, size, frames, steps):
    body = {
        "model": model, "prompt": prompt, "size": size, "seconds": 1,
        "nvext": {"num_frames": frames, "num_inference_steps": steps, "guidance_scale": 4.0},
    }
    t = time.perf_counter()
    try:
        async with session.post(url, json=body) as r:
            txt = await r.text()
            dt = time.perf_counter() - t
            ok = r.status == 200
            mb = 0.0
            if ok:
                try:
                    b64 = (json.loads(txt).get("data") or [{}])[0].get("b64_json") or ""
                    mb = len(b64) * 3 / 4 / 1e6
                except Exception:
                    ok = False
            return ok, dt, mb, r.status
    except Exception as e:
        return False, time.perf_counter() - t, 0.0, repr(e)[:40]


async def _worker(queue, session, url, results, args):
    while True:
        try:
            i, model = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        ok, dt, mb, st = await one_request(session, url, model, f"{args.prompt} #{i}",
                                           args.size, args.frames, args.steps)
        results.append((ok, dt, mb, model))
        short = model.split("/")[-1][:24]
        print(f"  req{i:03d} {short:24} {'OK ' if ok else 'ERR'} {dt:7.1f}s {mb:5.2f}MB http={st}", flush=True)
        queue.task_done()


async def main(args):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    url = args.url.rstrip("/") + "/v1/videos"
    queue = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait((i, models[i % len(models)]))
    results = []
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        await asyncio.gather(*[
            asyncio.create_task(_worker(queue, session, url, results, args))
            for _ in range(args.concurrency)
        ])
    wall = time.perf_counter() - t0
    oks = [r for r in results if r[0]]
    lat = sorted(d for ok, d, *_ in results if ok)
    print("\n==== /v1/videos benchmark (Dynamo + FlexTensor) ====", flush=True)
    print(f"  url={url}  models={len(models)}  concurrency={args.concurrency}  requests={args.requests}")
    print(f"  ok={len(oks)}/{len(results)}  errors={len(results) - len(oks)}")
    if lat:
        pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
        print(f"  gen latency: p50={statistics.median(lat):.1f}s  p99={pct(0.99):.1f}s  min={lat[0]:.1f}s  max={lat[-1]:.1f}s")
        print(f"  throughput: {len(oks) / wall:.3f} req/s = {len(oks) / wall * 60:.1f} videos/min  |  wall={wall:.1f}s")
        print("  reference: compare vs standalone warm gen (concurrency=1). Delta = Dynamo HTTP+serialize+route overhead.")
    print("====================================================", flush=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Dynamo /v1/videos benchmark")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--models", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="comma-separated served model names (round-robined)")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=6)
    p.add_argument("--prompt", default="a cat walking on green grass, cinematic")
    p.add_argument("--size", default="832x480")
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--timeout", type=int, default=1800)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
