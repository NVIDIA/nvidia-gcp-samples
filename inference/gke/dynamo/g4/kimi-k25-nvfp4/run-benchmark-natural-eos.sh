#!/usr/bin/env bash
# Natural-EOS benchmark for Kimi K2.5 NVFP4 -- matches Google reference methodology
# exactly (variable OSL, EOS-terminated). Use this when you need an apples-to-apples
# Dynamo-vs-Google throughput/TTFT comparison on the same workload Google ran.
#
# Difference vs run-benchmark.sh (locked OSL=8192, ignore_eos=true):
#   - aiperf:        drops --extra-inputs ignore_eos:true
#   - bench_serving: adds --disable-ignore-eos (this sglang version defaults ignore_eos=true)
#
# Expected (NVFP4, conc=512, 1536 reqs, ISL=1024, max OSL=8192 but natural EOS terminates):
#   - OSL achieved avg: ~4,189 tokens (matches Google's 6,434,886 total / 1,536 reqs)
#   - Duration: ~33 min (matches Google's 33.1 min)
#   - Throughput: 3,200-3,400 tok/s for Dynamo (~tie with Google's 3,237 standalone)
#   - TTFT P50: ~250-400 ms (matches Google's 304 ms standalone)
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
    URL="http://agg-sglang-kimi-k25-nvfp4-frontend.default.svc.cluster.local:8000"
    TAG="dynamo"
    ;;
  standalone)
    URL="http://kimi-k25-sglang-nvfp4-serving.default.svc.cluster.local:8000"
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

ARTIFACT_DIR=${ARTIFACT_DIR:-/workspace/results/kimi-k25-nvfp4-c512-1536req-${TAG}-natEOS-${HARNESS}}
mkdir -p "$ARTIFACT_DIR"

echo "==> Natural-EOS methodology (variable OSL, matches Google reference)"
echo "==> Target=${TARGET} (${URL}); Harness=${HARNESS}"
echo "==> Artifacts: ${ARTIFACT_DIR}"

if [[ "$HARNESS" == "aiperf" ]]; then
  # Random workload, ISL=1024, max OSL=8192 but no ignore_eos -> Kimi EOS terminates
  # at ~4,189 avg OSL (same as Google's reference run).
  aiperf profile \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --tokenizer nvidia/Kimi-K2.5-NVFP4 \
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
    --model nvidia/Kimi-K2.5-NVFP4 \
    --tokenizer nvidia/Kimi-K2.5-NVFP4 \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 8192 \
    --num-prompts 1536 \
    --max-concurrency 512 \
    --apply-chat-template \
    --disable-ignore-eos \
    --output-file "$ARTIFACT_DIR/bench_serving.json"
fi

echo "Results in: $ARTIFACT_DIR"
