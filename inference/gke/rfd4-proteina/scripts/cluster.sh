#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if ! gcloud compute networks describe "$NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud compute networks create "$NAME" --subnet-mode=custom --project="$PROJECT_ID"
fi
if ! gcloud compute networks subnets describe "$NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud compute networks subnets create "$NAME" --network="$NAME" --range=10.88.0.0/20 \
        --region="$REGION" --project="$PROJECT_ID"
fi
if ! gcloud container clusters describe "$NAME" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud container clusters create-auto "$NAME" --region="$REGION" --project="$PROJECT_ID" \
        --release-channel=regular \
        --network="$NAME" --subnetwork="$NAME" \
        --service-account="$NODE_SA" \
        --labels=sample=rfd4-proteina
fi
gcloud container clusters get-credentials "$NAME" --region="$REGION" --project="$PROJECT_ID"
