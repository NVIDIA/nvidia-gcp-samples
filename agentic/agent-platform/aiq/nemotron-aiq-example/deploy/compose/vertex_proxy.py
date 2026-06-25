#!/usr/bin/env python3
"""
Minimal proxy: accept OpenAI-format POST /v1/chat/completions and forward to Vertex :rawPredict.

Vertex custom endpoints (e.g. NIM) only support :rawPredict, not /chat/completions. The OpenAI
client used by shallow/deep researcher calls base_url + "/chat/completions", which returns 400
"Precondition check failed". This proxy accepts /chat/completions and forwards the body to
Vertex :rawPredict, then returns the response.

Token priority:
  1. Bearer token from the request (OPENAI_API_KEY passed by the OpenAI client)
  2. GCE metadata server (works in Docker on GCE without gcloud CLI)
  3. gcloud auth print-access-token (fallback for local dev)

Environment variables:
  REGION       - GCP region (e.g. us-central1)
  PROJECT      - GCP project ID
  ENDPOINT_ID  - Vertex AI endpoint ID
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
        token = auth[7:].strip()
        # Skip placeholder tokens — fall through to metadata server
        if token and token.lower() not in ("placeholder", "unused", "dummy"):
            return token
    return None


def get_token_metadata_server() -> str | None:
    """Get access token from GCE metadata server (works in Docker on GCE)."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1"
            "/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("access_token")
    except Exception:
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


def get_token(handler: BaseHTTPRequestHandler) -> str | None:
    return (
        get_token_from_request(handler)
        or get_token_metadata_server()
        or get_token_gcloud()
    )


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
    def do_GET(self):
        # Health check endpoint
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404, f"Not found: {self.path}. Use POST /v1/chat/completions")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        token = get_token(self)
        if not token:
            self.send_error(401, "No token available: checked request header, GCE metadata server, and gcloud CLI")
            return
        status, resp_body, content_type = forward_to_vertex(body, token)
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", len(resp_body))
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, format, *args):
        print(f"[vertex-proxy] {args[0]}", flush=True)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    url = get_vertex_raw_predict_url()
    server = HTTPServer(("", port), ProxyHandler)
    print(f"Vertex chat/completions proxy listening on http://0.0.0.0:{port}/v1", flush=True)
    print(f"Forwarding to: {url}", flush=True)
    print("Token sources: request Bearer → GCE metadata server → gcloud CLI", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
