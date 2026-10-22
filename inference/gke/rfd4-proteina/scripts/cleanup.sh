#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

# Autopilot can still be scaling nodes when workload cleanup completes.
delete_cluster() {
    local error_log attempt
    error_log=$(mktemp)
    for attempt in {1..15}; do
        if gcloud container clusters delete "$NAME" --region="$REGION" --project="$PROJECT_ID" --quiet 2>"$error_log"; then
            cat "$error_log" >&2
            rm -f "$error_log"
            return 0
        fi
        if ! grep -q 'incompatible operation' "$error_log" || [[ "$attempt" -eq 15 ]]; then
            cat "$error_log" >&2
            rm -f "$error_log"
            return 1
        fi
        echo "Waiting for the active GKE operation before retrying cluster deletion..." >&2
        sleep 20
    done
}

if gcloud container clusters describe "$NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud container clusters get-credentials "$NAME" --region="$REGION" --project="$PROJECT_ID"
    # Delete the claim while the cluster's CSI controller can reclaim its disk.
    pv_name=$(kubectl get pvc data -n "$NAME" --ignore-not-found -o jsonpath='{.spec.volumeName}')
    kubectl delete namespace "$NAME" --ignore-not-found --wait=true --timeout=300s
    if [[ -n "$pv_name" ]]; then
        kubectl wait --for=delete "pv/$pv_name" --timeout=300s
    fi
    delete_cluster
fi

if gcloud artifacts repositories describe "$NAME" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud artifacts repositories delete "$NAME" --location="$REGION" --project="$PROJECT_ID" --quiet
fi

if gcloud compute networks subnets describe "$NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud compute networks subnets delete "$NAME" --region="$REGION" --project="$PROJECT_ID" --quiet
fi

if gcloud compute networks describe "$NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud compute networks delete "$NAME" --project="$PROJECT_ID" --quiet
fi
