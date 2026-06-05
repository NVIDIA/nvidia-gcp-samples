#!/usr/bin/env bash
# Template (work in progress) — shared-prefix benchmark for Kimi K2.5 INT4 with
# configurable shared-prefix percentage. Pairs with the optimized Dynamo DGD
# (KV-aware router + radix cache).
#
# Default harness per target (override with HARNESS=...):
#   dynamo     -> aiperf         (radix-cache warmup methodology)
#   standalone -> bench_serving
#
# Shared-prefix percentage knob (env: SHARED_PERCENT, default 80):
#   - 98 (max shared) : 1024 prefix + 16 unique. ISL=1040.
#   - 80 (default)    : 820 prefix + 204 unique. ISL=1024.
#   - 50              : 512 prefix + 512 unique. ISL=1024.
#   - 0  (near-random): 1 + 1023.
#   - any other value: prefix = ISL*pct/100, unique = ISL - prefix.
#
# Usage:
#   ./run-benchmark-opt-shared.sh dynamo                         # 80% shared (default)
#   SHARED_PERCENT=98 ./run-benchmark-opt-shared.sh dynamo       # 98% shared
#   SHARED_PERCENT=80 HARNESS=bench ./run-benchmark-opt-shared.sh dynamo
#   ./run-benchmark-opt-shared.sh <full-url>                     # custom endpoint

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
    echo "Usage: $0 [dynamo|standalone|<url>]   (env: HARNESS=aiperf|bench  SHARED_PERCENT=80|98|50|0)" >&2
    exit 1
    ;;
esac

# Shared-prefix workload composition — total ISL ~1024 tokens, split prefix vs unique tail.
SHARED_PERCENT=${SHARED_PERCENT:-80}
TARGET_ISL=1024

case "$SHARED_PERCENT" in
  98) PREFIX_LEN=1024; SYNTHETIC=16 ;;     # max shared (ISL=1040)
  80) PREFIX_LEN=820;  SYNTHETIC=204 ;;
  50) PREFIX_LEN=512;  SYNTHETIC=512 ;;
  0)  PREFIX_LEN=1;    SYNTHETIC=1023 ;;   # near-random (aiperf requires prefix > 0)
  *)
    PREFIX_LEN=$(( TARGET_ISL * SHARED_PERCENT / 100 ))
    SYNTHETIC=$(( TARGET_ISL - PREFIX_LEN ))
    ;;
esac

POOL_SIZE=64
WARMUP_COUNT=$(( POOL_SIZE * 2 ))   # 2x pool so every prefix is warmed into radix cache

ARTIFACT_DIR=${ARTIFACT_DIR:-/workspace/results/kimi-k25-int4-c512-1536req-${TAG}-opt-${HARNESS}-shared${SHARED_PERCENT}}
mkdir -p "$ARTIFACT_DIR"

echo "==> Shared prefix=${SHARED_PERCENT}% (prefix=${PREFIX_LEN}, unique=${SYNTHETIC}); pool=${POOL_SIZE}; warmup=${WARMUP_COUNT}"
echo "==> Target=${TARGET} (${URL}); Harness=${HARNESS}"
echo "==> Artifacts: ${ARTIFACT_DIR}"

if [[ "$HARNESS" == "aiperf" ]]; then
  aiperf profile \
    --model moonshotai/Kimi-K2.5 \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --tokenizer moonshotai/Kimi-K2.5 \
    --tokenizer-trust-remote-code \
    --prefix-prompt-length "$PREFIX_LEN" \
    --prefix-prompt-pool-size "$POOL_SIZE" \
    --synthetic-input-tokens-mean "$SYNTHETIC" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 8192 --output-tokens-stddev 0 \
    --num-prompts 1536 --num-requests 1536 \
    --warmup-request-count "$WARMUP_COUNT" \
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
  HOST_PORT=${URL#http://}
  HOST_PORT=${HOST_PORT#https://}
  HOST=${HOST_PORT%:*}
  PORT=${HOST_PORT##*:}

  PROMPTS_PER_GROUP=$(( 1536 / POOL_SIZE ))   # 24 for pool=64
  python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$PORT" \
    --model moonshotai/Kimi-K2.5 \
    --tokenizer moonshotai/Kimi-K2.5 \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$POOL_SIZE" \
    --gsp-prompts-per-group "$PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len "$PREFIX_LEN" \
    --gsp-question-len "$SYNTHETIC" \
    --gsp-output-len 8192 \
    --max-concurrency 512 \
    --warmup-requests "$WARMUP_COUNT" \
    --apply-chat-template \
    --seed 100 \
    --extra-request-body '{"ignore_eos":true,"min_tokens":8192,"max_tokens":8192,"temperature":0.0,"repetition_penalty":1.0}' \
    --output-file "$ARTIFACT_DIR/bench_serving.json"
fi

echo "Results in: $ARTIFACT_DIR"
