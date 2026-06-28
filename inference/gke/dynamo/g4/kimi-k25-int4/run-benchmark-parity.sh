#!/usr/bin/env bash
# Clean Dynamo-vs-Standalone parity test (apples-to-apples for Goal 2).
#
# WHY THIS SCRIPT EXISTS:
# To answer "is Dynamo on par with Standalone SGLang at single-instance?" without
# methodology noise, we run the SAME aiperf command against both endpoints with:
#   - aiperf on both sides       (no harness instrumentation drift)
#   - LOCKED OSL=8192 (ignore_eos=true + min/max_tokens=8192)
#                                (eliminates EOS-distribution variance; engine
#                                 runs in steady-state decode where Dynamo
#                                 overhead is most visible)
#   - random workload            (no shared prefix -- KV routing has no effect)
#   - deterministic sampling     (temperature=0, rep_penalty=1.0, random-seed=100)
#   - same engine config         (Standalone manifest = Dynamo DGD engine flags)
# Any leftover delta is Dynamo's Frontend / service-mesh / router overhead.
#
# NOTE on methodology choice (locked vs variable OSL):
#   For Goal 2 (Dynamo-vs-Standalone parity on OUR cluster), LOCKED OSL is the
#   cleaner test -- removes EOS variance, makes overhead visible.
#   For Goal 1 (Standalone vs Google reference), use VARIABLE OSL via
#   `run-benchmark-natural-eos.sh standalone` -- matches Google's methodology.
#
# PREREQUISITES:
#   1. Dynamo parity DGD applied (random router):
#      kubectl apply -f dgd-agg-sglang-kimi-k25-int4-parity.yaml
#      (Uses --router-mode random; KV routing has no benefit at replicas=1.)
#   2. Standalone INT4 deployment ALSO running for the standalone leg:
#      kubectl apply -f standalone-sglang-kimi-k25-int4.yaml
#      (They share the model-cache PVC; OK to have both deployed simultaneously.)
#   3. Perf pod running:
#      kubectl apply -f benchmark-kimi-k25-int4-pod.yaml
#
# USAGE:
#   ./run-benchmark-parity.sh standalone                 # aiperf vs Standalone (default)
#   ./run-benchmark-parity.sh dynamo                     # aiperf vs Dynamo (default)
#   HARNESS=bench  ./run-benchmark-parity.sh dynamo      # bench_serving cross-check vs Dynamo
#   HARNESS=bench  ./run-benchmark-parity.sh standalone  # bench_serving cross-check vs Standalone
#   HARNESS=aiperf ./run-benchmark-parity.sh dynamo      # explicit aiperf (same as default)
#   ./run-benchmark-parity.sh <full-url>                 # custom endpoint
#
# Run both targets sequentially (don't overlap on the same cluster).
#
# Note: this locked-OSL aiperf run isolates Dynamo-wrapper overhead vs Standalone
# at identical workload — it is NOT methodology-matched to Google's published number.
# For the Google-comparable run see run-benchmark-natural-eos.sh standalone.

set -euo pipefail

TARGET=${1:-dynamo}
HARNESS=${HARNESS:-aiperf}      # default aiperf; override with HARNESS=bench for sglang.bench_serving

case "$TARGET" in
  dynamo)
    URL="http://agg-sglang-kimi-k25-int4-frontend.default.svc.cluster.local:8000"
    TAG="dynamo"
    ;;
  standalone)
    URL="http://kimi-k25-sglang-serving.default.svc.cluster.local:8000"
    TAG="standalone"
    ;;
  http*)
    URL="$TARGET"
    TAG="custom"
    ;;
  *)
    echo "Usage: $0 [dynamo|standalone|<url>]   (env: HARNESS=aiperf|bench)" >&2
    exit 1
    ;;
esac

ARTIFACT_DIR=${ARTIFACT_DIR:-/workspace/results/kimi-k25-int4-c512-1536req-${TAG}-parity-${HARNESS}}
mkdir -p "$ARTIFACT_DIR"

echo "==> Parity test (${HARNESS}, LOCKED OSL=8192, random workload, deterministic sampling)"
echo "==> Target=${TARGET} (${URL})"
echo "==> Artifacts: ${ARTIFACT_DIR}"

# aiperf with:
#   - ISL=1024 (locked synthetic input length)
#   - OSL=8192 LOCKED via ignore_eos + min_tokens=max_tokens=8192
#       (every request generates exactly 8192 tokens -- removes EOS variance,
#        engine runs in steady-state decode, Dynamo overhead most visible)
#   - random prompts (no --prefix-prompt-* flags) -> no shared prefix
#   - conc=512, 1536 requests -> matches Google reference workload sizing
#   - Deterministic sampling (temperature=0, rep_penalty=1.0) + random-seed=100
#     -> reproducible across runs
if [[ "$HARNESS" == "aiperf" ]]; then
  aiperf profile \
    --model moonshotai/Kimi-K2.5 \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --tokenizer moonshotai/Kimi-K2.5 \
    --tokenizer-trust-remote-code \
    --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 8192 --output-tokens-stddev 0 \
    --num-prompts 1536 --num-requests 1536 \
    --use-server-token-count \
    --concurrency 512 \
    --record-processors 16 \
    --random-seed 100 \
    --extra-inputs ignore_eos:true \
    --extra-inputs max_tokens:8192 \
    --extra-inputs min_tokens:8192 \
    --extra-inputs temperature:0.0 \
    --extra-inputs repetition_penalty:1.0 \
    --artifact-dir "$ARTIFACT_DIR" \
    --ui simple
else
  # bench_serving cross-check path -- random workload, locked OSL=8192, deterministic sampling.
  # Mirrors the aiperf path's methodology for harness cross-validation.
  HOST_PORT=${URL#http://}
  HOST_PORT=${HOST_PORT#https://}
  HOST=${HOST_PORT%:*}
  PORT=${HOST_PORT##*:}

  python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$PORT" \
    --model moonshotai/Kimi-K2.5 \
    --tokenizer moonshotai/Kimi-K2.5 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 8192 \
    --num-prompts 1536 \
    --max-concurrency 512 \
    --apply-chat-template \
    --seed 100 \
    --extra-request-body '{"ignore_eos":true,"min_tokens":8192,"max_tokens":8192,"temperature":0.0,"repetition_penalty":1.0}' \
    --output-file "$ARTIFACT_DIR/bench_serving.json"
fi

echo "Results in: $ARTIFACT_DIR"
