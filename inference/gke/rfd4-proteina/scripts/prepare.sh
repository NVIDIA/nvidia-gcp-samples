#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
# Enable only missing APIs, so pre-enabled projects need no enable permission.
enabled=$(gcloud services list --enabled --project="$PROJECT_ID" --format='value(config.name)')
for api in container compute artifactregistry cloudbuild storage iam iamcredentials; do
    if ! echo "$enabled" | rg -q "^${api}\.googleapis\.com$"; then
        gcloud services enable "${api}.googleapis.com" --project="$PROJECT_ID"
    fi
done
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://$BUCKET" --project="$PROJECT_ID" \
        --location="$REGION" --uniform-bucket-level-access --public-access-prevention
fi
if ! gcloud artifacts repositories describe "$NAME" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$NAME" --repository-format=docker \
        --location="$REGION" --project="$PROJECT_ID"
fi
# Build/node accounts must already have the documented project permissions.
echo "Build staging bucket and image repository ready."
