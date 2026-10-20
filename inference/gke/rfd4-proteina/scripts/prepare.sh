#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

# Enable only missing APIs in a single batched call, so pre-enabled projects need no enable permission.
enabled=$(gcloud services list --enabled --project="$PROJECT_ID" --format='value(config.name)')
missing_apis=()
for api in container compute artifactregistry cloudbuild storage iam iamcredentials logging monitoring; do
    if ! echo "$enabled" | grep -q "^${api}\.googleapis\.com$"; then
        missing_apis+=("${api}.googleapis.com")
    fi
done
if (( ${#missing_apis[@]} > 0 )); then
    gcloud services enable "${missing_apis[@]}" --project="$PROJECT_ID"
fi

if ! gcloud artifacts repositories describe "$NAME" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$NAME" --repository-format=docker \
        --location="$REGION" --project="$PROJECT_ID"
fi

# Newly created service accounts can take a few seconds to propagate in IAM
# before projects.add-iam-policy-binding recognizes them.
bind_role() {
    local sa="$1" role="$2"
    for attempt in 1 2 3 4 5; do
        if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${sa}" --role="$role" --condition=None --quiet >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${sa}" --role="$role" --condition=None --quiet >/dev/null
}

# Create dedicated, least-privilege build and node service accounts if they do not exist.
if [[ "$BUILD_SA" == "${NAME}-build@${PROJECT_ID}.iam.gserviceaccount.com" ]]; then
    if ! gcloud iam service-accounts describe "$BUILD_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
        gcloud iam service-accounts create "${NAME}-build" \
            --display-name="RFD4-Proteina Cloud Build SA" --project="$PROJECT_ID"
    fi
    for role in roles/logging.logWriter roles/storage.admin roles/artifactregistry.writer; do
        bind_role "$BUILD_SA" "$role"
    done
fi

if [[ "$NODE_SA" == "${NAME}-node@${PROJECT_ID}.iam.gserviceaccount.com" ]]; then
    if ! gcloud iam service-accounts describe "$NODE_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
        gcloud iam service-accounts create "${NAME}-node" \
            --display-name="RFD4-Proteina GKE Node SA" --project="$PROJECT_ID"
    fi
    for role in roles/container.defaultNodeServiceAccount roles/artifactregistry.reader; do
        bind_role "$NODE_SA" "$role"
    done
fi

echo "Image repository and service accounts ready."
