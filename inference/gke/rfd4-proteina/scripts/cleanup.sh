#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

if gcloud container clusters describe "$NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud container clusters get-credentials "$NAME" --region="$REGION" --project="$PROJECT_ID"
    # Delete the claim while the cluster's CSI controller can reclaim its disk.
    pv_name=$(kubectl get pvc data -n "$NAME" --ignore-not-found -o jsonpath='{.spec.volumeName}')
    kubectl delete namespace "$NAME" --ignore-not-found --wait=true --timeout=300s
    if [[ -n "$pv_name" ]]; then
        kubectl wait --for=delete "pv/$pv_name" --timeout=300s
    fi
    gcloud container clusters delete "$NAME" --region="$REGION" --project="$PROJECT_ID" --quiet
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
