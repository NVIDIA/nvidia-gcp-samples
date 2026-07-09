#!/usr/bin/env bash
# Functional validation for GLM-5.2-NVFP4 on SGLang (SM120 recipe).
# Usage: ./smoke-test.sh <host> <port> [chat|generate]
#   generate = SGLang native completion endpoint (standalone, port 30000)
#   chat     = OpenAI-compatible chat endpoint (Dynamo frontend port 8000, or standalone /v1)
# All checks run at temperature 0. Exit code 0 = all passed.
set -u
HOST="${1:-localhost}"
PORT="${2:-30000}"
MODE="${3:-generate}"
MODEL="nvidia/GLM-5.2-NVFP4"
PASS=0; FAIL=0

ask_chat() {  # $1=prompt $2=max_tokens
  curl -s "http://${HOST}:${PORT}/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": $(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
    \"max_tokens\": $2, \"temperature\": 0}" |
  python3 -c 'import json,sys
d = json.load(sys.stdin); m = d["choices"][0]["message"]
print((m.get("content") or "") + " " + (m.get("reasoning_content") or ""))'
}

ask_gen() {  # $1=prompt $2=max_tokens
  curl -s "http://${HOST}:${PORT}/generate" -H "Content-Type: application/json" -d "{
    \"text\": $(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"sampling_params\": {\"max_new_tokens\": $2, \"temperature\": 0}}" |
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("text",""))'
}

check() {  # $1=name $2=expected-substring $3=response
  if printf '%s' "$3" | grep -qi -- "$2"; then
    echo "PASS  $1"; PASS=$((PASS+1))
  else
    echo "FAIL  $1  (expected to contain: $2)"
    echo "      got: $(printf '%s' "$3" | tr '\n' ' ' | cut -c1-160)"; FAIL=$((FAIL+1))
  fi
}

echo "== GLM-5.2-NVFP4 smoke test  host=$HOST port=$PORT mode=$MODE =="

NEEDLE_DOC=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 400)')

if [ "$MODE" = "chat" ]; then
  # chat endpoint: instruction-following prompts (validated form)
  R=$(ask_chat "Reply with exactly READY." 600);                                    check "exact instruction (READY)" "READY" "$R"
  R=$(ask_chat "What is the capital of France? Answer in one short sentence." 600); check "factual QA (Paris)" "Paris" "$R"
  R=$(ask_chat "What is 17 + 25? Answer with just the number." 600);                check "arithmetic (42)" "42" "$R"
  R=$(ask_chat "$NEEDLE_DOC
Question: What animal jumps over the dog in the text above? Answer in one word." 600)
  check "long-context needle, ~3600 tokens (fox)" "fox" "$R"
else
  # raw completion endpoint: continuation-style prompts (validated form)
  R=$(ask_gen "The capital of France is" 24);                    check "factual completion (Paris)" "Paris" "$R"
  R=$(ask_gen "Question: What is 17 + 25?
Answer:" 24);                                                    check "arithmetic completion (42)" "42" "$R"
  R=$(ask_gen "The chemical symbol for gold is" 24);             check "factual completion (Au)" "Au" "$R"
  R=$(ask_gen "$NEEDLE_DOC
Question: What animal jumps over the dog in the text above? Answer in one word:" 24)
  check "long-context needle, ~3600 tokens (fox)" "fox" "$R"
fi

echo "== result: ${PASS} passed, ${FAIL} failed =="
[ "$FAIL" -eq 0 ] || exit 1
