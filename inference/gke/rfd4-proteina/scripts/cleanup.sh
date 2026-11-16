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

# Successful list calls distinguish absent resources from permission or API failures.
# Keep these assignments outside conditionals so set -e stops on lookup errors.
cluster=$(gcloud container clusters list --location="$REGION" --project="$PROJECT_ID" \
    --filter="name=${NAME}" --format='value(name)')
if [[ -n "$cluster" ]]; then
    gcloud container clusters get-credentials "$NAME" --region="$REGION" --project="$PROJECT_ID"
    # Claim references remain on terminating PVs after their PVC/namespace is gone.
    # Discover them on every run so a retry still waits for disk reclamation.
    pv_names=$(kubectl get pv --no-headers \
        -o custom-columns=NAME:.metadata.name,NAMESPACE:.spec.claimRef.namespace \
        | awk -v namespace="$NAME" '$2 == namespace {print $1}')
    kubectl delete namespace "$NAME" --ignore-not-found --wait=true --timeout=300s
    while IFS= read -r pv_name; do
        [[ -n "$pv_name" ]] || continue
        kubectl wait --for=delete "pv/$pv_name" --timeout=300s
    done <<< "$pv_names"
    delete_cluster
fi

repository=$(gcloud artifacts repositories list --location="$REGION" --project="$PROJECT_ID" \
    --filter="name=projects/${PROJECT_ID}/locations/${REGION}/repositories/${NAME}" --format='value(name)')
if [[ -n "$repository" ]]; then
    gcloud artifacts repositories delete "$NAME" --location="$REGION" --project="$PROJECT_ID" --quiet
fi

subnet=$(gcloud compute networks subnets list --regions="$REGION" --project="$PROJECT_ID" \
    --filter="name=${NAME}" --format='value(name)')
if [[ -n "$subnet" ]]; then
    gcloud compute networks subnets delete "$NAME" --region="$REGION" --project="$PROJECT_ID" --quiet
fi

network=$(gcloud compute networks list --project="$PROJECT_ID" \
    --filter="name=${NAME}" --format='value(name)')
if [[ -n "$network" ]]; then
    gcloud compute networks delete "$NAME" --project="$PROJECT_ID" --quiet
fi
