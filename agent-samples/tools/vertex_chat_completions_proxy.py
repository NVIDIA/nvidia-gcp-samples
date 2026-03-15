#!/usr/bin/env python3
"""
Minimal proxy: accept OpenAI-format POST /v1/chat/completions and forward to Vertex :rawPredict.

Vertex custom endpoints (e.g. NIM) only support :rawPredict, not /chat/completions. The OpenAI
client used by shallow/deep researcher calls base_url + "/chat/completions", which returns 400
"Precondition check failed". This proxy accepts /chat/completions and forwards the body to
Vertex :rawPredict, then returns the response.

Usage:
  1. Set REGION, PROJECT, ENDPOINT_ID in the environment (same as your notebook).
  2. Run the proxy on the same host as the Jupyter kernel (so localhost is reachable):
       python vertex_chat_completions_proxy.py [port]
     Default port 8765.
  3. In the notebook, set ENDPOINT_URL = "http://localhost:8765/v1" and OPENAI_API_KEY to your
     gcloud token (see notebook_env_setup_for_vertex.py), then run the workflow.

The proxy uses the Bearer token from the request (OPENAI_API_KEY) to call Vertex; if missing,
it falls back to gcloud auth print-access-token.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


def get_vertex_raw_predict_url() -> str:
    region = os.environ.get("REGION")
    project = os.environ.get("PROJECT")
    endpoint_id = os.environ.get("ENDPOINT_ID")
    return (
        f"https://{region}-prediction-aiplatform.googleapis.com/v1"
        f"/projects/{project}/locations/{region}/endpoints/{endpoint_id}:rawPredict"
    )


def get_token_from_request(handler: BaseHTTPRequestHandler) -> str | None:
    auth = handler.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def get_token_gcloud() -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def forward_to_vertex(body: bytes, token: str) -> tuple[int, bytes, str]:
    url = get_vertex_raw_predict_url()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "application/json")


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404, f"Not found: {self.path}. Use POST /v1/chat/completions")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        token = get_token_from_request(self) or get_token_gcloud()
        if not token:
            self.send_error(401, "Missing Authorization Bearer token and gcloud token unavailable")
            return
        status, resp_body, content_type = forward_to_vertex(body, token)
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", len(resp_body))
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}", flush=True)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("", port), ProxyHandler)
    print(f"Vertex chat/completions proxy listening on http://localhost:{port}/v1", flush=True)
    print("Set ENDPOINT_URL=http://localhost:{}/v1 and run your workflow.".format(port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
