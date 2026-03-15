"""
Vertex AI rawPredict client for OpenAI-format chat completions.
Use this when your Vertex endpoint is a custom container (e.g. NIM) that expects
the request at :rawPredict with no /chat/completions path.

Drop into aiq (e.g. aiq_agent/llm/vertex_raw_predict_client.py) and use
get_vertex_raw_predict_client() when config has use_vertex_raw_predict=True.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Configure from env or override in code (env: REGION, PROJECT, ENDPOINT_ID, RAW_PREDICT_BASE_URL, MODEL, USE_ADC)
REGION = os.environ.get("REGION")
PROJECT = os.environ.get("PROJECT")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID")
# Optional: full URL overrides region/project/endpoint_id if set
RAW_PREDICT_BASE_URL = os.environ.get("RAW_PREDICT_BASE_URL", "").strip()
# Model name sent in the request body (NIM/custom endpoints often require a specific id)
MODEL = os.environ.get("MODEL", "").strip() or "default"


def get_raw_predict_url(
    *,
    region: str | None = None,
    project: str | None = None,
    endpoint_id: str | None = None,
) -> str:
    """Build the Vertex rawPredict URL (no path after :rawPredict)."""
    region = region or REGION
    project = project or PROJECT
    endpoint_id = endpoint_id or ENDPOINT_ID
    return (
        f"https://{region}-prediction-aiplatform.googleapis.com/v1"
        f"/projects/{project}/locations/{region}/endpoints/{endpoint_id}:rawPredict"
    )


async def chat_completions_create(
    messages: list[dict[str, Any]],
    *,
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = False,
    base_url: str | None = None,
    **kwargs: Any,
) -> str | Any:
    """
    Call Vertex endpoint via rawPredict with OpenAI-format chat body.
    URL is base_url or get_raw_predict_url(); no /chat/completions is appended.
    Returns the content string for non-stream, or the response for stream=True.
    """
    url = base_url or RAW_PREDICT_BASE_URL or get_raw_predict_url()
    model_id = model if model != "default" else MODEL

    # Prefer gcloud token (same as curl). Set USE_ADC=1 to force ADC only.
    token = None
    token_source = "none"
    if os.environ.get("USE_ADC", "").strip().lower() not in ("1", "true", "yes"):
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                token = result.stdout.strip()
                token_source = "gcloud"
        except Exception:
            pass
    if not token:
        import google.auth
        import google.auth.transport.requests
        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())
        token = credentials.token
        token_source = "ADC"
    logger.info("Vertex rawPredict auth: %s, URL: %s", token_source, url)

    body = {
        "model": model_id,
        "messages": [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        **{k: v for k, v in kwargs.items() if v is not None},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if r.status_code == 404:
                body = (r.text or "").strip() or "(empty)"
                logger.error(
                    "Vertex rawPredict 404. Auth used: %s. Response body: %s",
                    token_source,
                    body[:500] if len(body) > 500 else body,
                )
                if "does not exist" in body and "model" in body.lower():
                    raise ValueError(
                        "The model '%s' does not exist on this endpoint. Set MODEL (env) to the model ID "
                        "your endpoint expects (e.g. the NIM/deployed model name). Response: %s"
                        % (model_id, body[:300])
                    ) from e
                raise ValueError(
                    "Vertex endpoint not found (404). Check the log above for auth source and response body. "
                    "If curl works but this fails, the process may be using different credentials. "
                    "URL: %s" % url
                ) from e
            raise
        data = r.json()

    if stream:
        return data

    # Return a simple object so callers can use response.choices[0].message.content
    class _Choice:
        pass

    class _Message:
        content: str = ""

    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        m = _Message()
        m.content = msg.get("content", "") or ""
        c = _Choice()
        c.message = m
        return type("Response", (), {"choices": [c], "raw": data})()
    # Fallback
    c = _Choice()
    c.message = _Message()
    c.message.content = data.get("text", str(data))
    return type("Response", (), {"choices": [c], "raw": data})()


def get_vertex_raw_predict_client():
    """
    Return a small object that mimics OpenAI client's chat.completions.create
    for use in aiq when use_vertex_raw_predict is True.
    """
    class VertexRawPredictChat:
        @staticmethod
        async def create(
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> str | Any:
            return await chat_completions_create(messages, **kwargs)

    return VertexRawPredictChat()
