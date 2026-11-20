#!/usr/bin/env python3
"""Submit the upstream monomer example, wait for completion, and download its CIF."""

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import zstandard

SAMPLE = Path(__file__).resolve().parents[1]


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = raw[:500]
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc


@contextmanager
def forward(namespace, port):
    # Refuse occupied ports rather than accidentally querying a different service.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is in use; choose another with --port.") from exc
    log = SAMPLE / ".build" / "example-port-forward.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(
            ["kubectl", "port-forward", "--address=127.0.0.1", "-n", namespace, "service/api", f"{port}:8000"],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            yield process, log
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run(args):
    source = SAMPLE / "upstream/data/benchmarks/monomer/monomer-short.cif.zst"
    content = zstandard.ZstdDecompressor().decompress(source.read_bytes()).decode()
    payload = {
        "task": "generation",
        "input": {"format": "annotated_cif", "file": {"content": content}},
        "sampling": {"seed": 42, "num_replicates": 1},
    }
    base = f"http://127.0.0.1:{args.port}"
    with forward(args.namespace, args.port) as (process, log):
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Port forwarding stopped; see {log}.")
            try:
                ready = json.loads(request(base, "/health/ready"))
                if ready.get("status") == "ready":
                    break
            except (OSError, RuntimeError):
                pass  # The local tunnel may take a moment to start.
            if time.monotonic() >= deadline:
                raise TimeoutError(f"API did not become ready; see {log} and kubectl logs.")
            time.sleep(1)

        # Submit exactly once: retrying a POST after a dropped connection can duplicate work.
        job = json.loads(request(base, "/v1/designs", payload))
        request_id = job["id"]
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise RuntimeError("API returned an invalid request ID.")
        output = SAMPLE / "results" / request_id
        output.mkdir(parents=True, exist_ok=True)
        print(
            f"Submitted {request_id}. Waiting for generation (cold compilation can take several minutes).", flush=True
        )
        deadline = time.monotonic() + args.timeout
        previous = None
        while True:
            (output / "status.json").write_text(json.dumps(job, indent=2))
            if job["status"] != previous:
                print(f"Status: {job['status']}", flush=True)
                previous = job["status"]
            if job["status"] == "succeeded":
                break
            if job["status"] == "failed":
                raise RuntimeError(f"Design {request_id} failed: {json.dumps(job.get('error'))}")
            if process.poll() is not None:
                raise RuntimeError(f"Port forwarding stopped; request {request_id} may still be running. See {log}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {request_id}; its last status is saved in {output}.")
            time.sleep(min(5, max(0, deadline - time.monotonic())))
            job = json.loads(request(base, f"/v1/designs/{request_id}"))

        listing = json.loads(request(base, f"/v1/designs/{request_id}/results"))
        generated = next((a for a in listing["artifacts"] if a["name"].endswith("_generated.cif")), None)
        if generated is None:
            raise RuntimeError("The completed design has no generated CIF artifact.")
        download = generated["download"]
        if not download.startswith(f"/v1/designs/{request_id}/artifacts/"):
            raise RuntimeError("API returned an unexpected artifact URL.")
        cif = request(base, download)
        if not cif:
            raise RuntimeError("The downloaded CIF is empty.")
        (output / "design.cif").write_bytes(cif)
        print(f"Downloaded {output / 'design.cif'} ({len(cif)} bytes).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=os.environ.get("NAME", "rfd4-proteina"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--timeout", type=float, default=3600, help="Maximum generation wait in seconds (default: 3600)"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error("--port must be 1–65535 and --timeout must be positive")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped waiting; an accepted design request may still be running.\n")
