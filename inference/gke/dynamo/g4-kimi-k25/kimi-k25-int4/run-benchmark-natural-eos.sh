#!/usr/bin/env bash
# Natural-EOS benchmark for Kimi K2.5 INT4 — matches Google reference methodology
# (variable OSL via natural EOS termination). Use for the direct Google comparison.
#
# Difference vs run-benchmark-parity.sh (locked OSL=8192):
#   - aiperf:        drops --extra-inputs ignore_eos:true
#   - bench_serving: adds --disable-ignore-eos
#
# Default harness per target (override with HARNESS=...):
#   standalone -> bench_serving (matches Google reference command exactly)
#   dynamo     -> aiperf        (Dynamo customer-facing harness)
#
# Usage:
#   ./run-benchmark-natural-eos.sh standalone                  # Google-methodology standalone
#   ./run-benchmark-natural-eos.sh dynamo                      # Dynamo at Google methodology
#   HARNESS=bench ./run-benchmark-natural-eos.sh dynamo        # bench cross-check on Dynamo
#   ./run-benchmark-natural-eos.sh <full-url>

set -euo pipefail

TARGET=${1:-dynamo}
case "$TARGET" in
  standalone) DEFAULT_HARNESS=bench ;;
  *)          DEFAULT_HARNESS=aiperf ;;
esac
HARNESS=${HARNESS:-$DEFAULT_HARNESS}

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

ARTIFACT_DIR=${ARTIFACT_DIR:-/workspace/results/kimi-k25-int4-c512-1536req-${TAG}-natEOS-${HARNESS}}
mkdir -p "$ARTIFACT_DIR"

echo "==> Natural-EOS methodology (variable OSL, matches Google reference)"
echo "==> Target=${TARGET} (${URL}); Harness=${HARNESS}"
echo "==> Artifacts: ${ARTIFACT_DIR}"

if [[ "$HARNESS" == "aiperf" ]]; then
  # Random workload, ISL=1024, max OSL=8192 but no ignore_eos -> Kimi EOS terminates
  # at ~4,189 avg OSL (same as Google's reference run).
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
    --concurrency 512 \
    --artifact-dir "$ARTIFACT_DIR" \
    --ui simple
else
  HOST_PORT=${URL#http://}
  HOST_PORT=${HOST_PORT#https://}
  HOST=${HOST_PORT%:*}
  PORT=${HOST_PORT##*:}

  # bench_serving's default in this sglang version is ignore_eos=true; explicitly
  # disable it to match Google's natural-EOS methodology.
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
    --disable-ignore-eos \
    --output-file "$ARTIFACT_DIR/bench_serving.json"
fi

echo "Results in: $ARTIFACT_DIR"
