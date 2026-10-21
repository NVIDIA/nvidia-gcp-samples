#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
gcloud container clusters get-credentials "$NAME" --region="$REGION" --project="$PROJECT_ID"
digest=$(gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT_ID" --format='value(image_summary.digest)')
export DEPLOY_IMAGE="${IMAGE%:*}@${digest}"
mkdir -p .build
kubectl kustomize overlay \
    | sed -e "s|namespace: rfd4-proteina|namespace: ${NAME}|g" \
          -e "s|  name: rfd4-proteina|  name: ${NAME}|g" \
          -e "s|image: rfd4-api:local|image: ${DEPLOY_IMAGE}|g" \
    > .build/api.yaml
kubectl apply -f .build/api.yaml
kubectl rollout status deployment/api -n "$NAME" --timeout=1800s
