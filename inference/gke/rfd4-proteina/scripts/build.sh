#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RFD4_SOURCE="${RFD4_SOURCE:-$PWD/upstream}"
if [[ ! -d "$RFD4_SOURCE/.git" ]]; then
    git clone "$RFD4_REPO" "$RFD4_SOURCE"
fi
if ! git -C "$RFD4_SOURCE" cat-file -e "${RFD4_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$RFD4_SOURCE" fetch "$RFD4_REPO" "$RFD4_COMMIT"
fi
git -C "$RFD4_SOURCE" checkout --detach "$RFD4_COMMIT"
if [[ ! -f "$RFD4_SOURCE/Dockerfile" || ! -f "$RFD4_SOURCE/deploy/kubernetes/kustomization.yaml" ]]; then
    echo "The selected commit must include the portable deployment assets." >&2
    exit 1
fi
mkdir -p .build
gcloud builds submit "$RFD4_SOURCE" \
    --project="$PROJECT_ID" \
    --config=cloudbuild.yaml \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
    --substitutions="_IMAGE=$IMAGE" \
    --async --format='value(id)' | tee .build/build-id
echo "Build submitted; follow it with: gcloud beta builds log $(cat .build/build-id) --stream --project=$PROJECT_ID"
