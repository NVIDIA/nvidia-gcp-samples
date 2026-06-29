#!/usr/bin/env bash
# KV-aware routing benchmark — concurrency sweep.
#
# Sweeps concurrency (4 → 8 → 16 → 32) with a fixed shared-prefix workload
# to show Dynamo's throughput and TTFT advantage over standalone SGLang
# on memory-constrained hardware (Kimi K2.5 NVFP4, TP=8, ~2.6 GB KV headroom).
#
# With 32 distinct prefixes and random routing (standalone), each replica
# must cache all 32 prefixes — wasting scarce KV memory on duplicates.
# As concurrency rises, KV pool overflows → evictions → TTFT spikes / OOM.
#
# With KV-aware routing (Dynamo), same-prefix requests are steered to the
# same replica → each replica caches only ~16 prefixes → more KV room for
# active requests → higher sustainable concurrency, lower TTFT.
#
# Prerequisites:
#   For "standalone": kubectl apply -f standalone-sglang-kimi-k25-nvfp4.yaml
#   For "dynamo":     kubectl apply -f dgd-kv-routing-experiment.yaml
#   Perf pod:         kubectl apply -f benchmark-kimi-k25-nvfp4-pod.yaml
#   Copy this script: kubectl cp run-kv-routing-sweep.sh perf-kimi-k25-nvfp4:/workspace/
#
# Usage:
#   ./run-kv-routing-sweep.sh standalone
#   ./run-kv-routing-sweep.sh dynamo
#   CONCURRENCIES="4 16" ./run-kv-routing-sweep.sh standalone
#   ./run-kv-routing-sweep.sh <full-url>

set -euo pipefail

TARGET=${1:-dynamo}

case "$TARGET" in
  standalone)
    URL="http://standalone-sglang-kimi-k25-frontend.default.svc.cluster.local:8000"
    TAG="standalone"
    ;;
  dynamo)
    URL="http://agg-sglang-kimi-k25-nvfp4-frontend.default.svc.cluster.local:8000"
    TAG="dynamo"
    ;;
  http*)
    URL="$TARGET"
    TAG="custom"
    ;;
  *)
    echo "Usage: $0 [standalone|dynamo|<url>]" >&2
    echo "  env: CONCURRENCIES='4 8 16 32'  POOL_SIZE=32  NUM_REQUESTS=64" >&2
    exit 1
    ;;
esac

MODEL=nvidia/Kimi-K2.5-NVFP4
PREFIX_LEN=820         # 80% of ISL ~1024
SYNTHETIC=204          # 20% unique tail per request
OSL=8192               # locked via ignore_eos + min/max tokens
POOL_SIZE=${POOL_SIZE:-32}
WARMUP=${WARMUP:-32}
NUM_REQUESTS=${NUM_REQUESTS:-64}

read -ra CONCS <<< "${CONCURRENCIES:-4 8 16 32}"

BASE_DIR=${ARTIFACT_DIR:-/workspace/results/kv-routing-conc-sweep}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "================================================================"
echo " KV Routing Concurrency Sweep"
echo " Target:       ${TAG}"
echo " Endpoint:     ${URL}"
echo " Concurrencies: ${CONCS[*]}"
echo " Pool size:    ${POOL_SIZE} (fixed)"
echo " Prefix:       ${PREFIX_LEN} tokens (80%), Unique: ${SYNTHETIC} tokens"
echo " OSL:          ${OSL} (locked, ignore_eos=true)"
echo " Warmup:       ${WARMUP} requests"
echo " Requests:     ${NUM_REQUESTS} per concurrency point"
echo " Timestamp:    ${TIMESTAMP}"
echo "================================================================"

for CONC in "${CONCS[@]}"; do
  RUN_DIR="${BASE_DIR}/${TAG}-conc${CONC}-${TIMESTAMP}"
  mkdir -p "$RUN_DIR"

  echo ""
  echo "------------------------------------------------------------"
  echo ">>> [${TAG}] Concurrency=${CONC}, pool=${POOL_SIZE}, warmup=${WARMUP}"
  echo ">>> Artifacts: ${RUN_DIR}"
  echo "------------------------------------------------------------"

  aiperf profile \
    --model "$MODEL" \
    --url "$URL" \
    --endpoint-type chat \
    --streaming \
    --tokenizer "$MODEL" \
    --tokenizer-trust-remote-code \
    --prefix-prompt-length "$PREFIX_LEN" \
    --num-prefix-prompts "$POOL_SIZE" \
    --synthetic-input-tokens-mean "$SYNTHETIC" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
    --num-prompts "$NUM_REQUESTS" --num-requests "$NUM_REQUESTS" \
    --warmup-request-count "$WARMUP" \
    --use-server-token-count \
    --concurrency "$CONC" \
    --record-processors 16 \
    --random-seed 100 \
    --extra-inputs ignore_eos:true \
    --extra-inputs max_tokens:${OSL} \
    --extra-inputs min_tokens:${OSL} \
    --extra-inputs temperature:0.0 \
    --extra-inputs repetition_penalty:1.0 \
    --artifact-dir "$RUN_DIR" \
    --ui simple \
    2>&1 | tee "${RUN_DIR}/aiperf.log"

  echo ">>> [${TAG}] Concurrency ${CONC} complete"
  echo ">>> Results: ${RUN_DIR}"
done

echo ""
echo "================================================================"
echo " Sweep complete: ${TAG}"
echo " All results under: ${BASE_DIR}/"
echo ""
echo " Next steps:"
echo "   1. If you just ran 'standalone', deploy Dynamo and run 'dynamo'"
echo "   2. Compare TTFT p50 and throughput across concurrency levels"
echo "   3. Expected: Dynamo sustains higher concurrency with lower TTFT"
echo "================================================================"
