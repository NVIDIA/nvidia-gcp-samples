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
if ! git -C "$RFD4_SOURCE" cat-file -e "${RFD4_COMMIT}:Dockerfile" || ! git -C "$RFD4_SOURCE" cat-file -e "${RFD4_COMMIT}:deploy/kubernetes/kustomization.yaml"; then
    echo "The selected commit must include the portable deployment assets." >&2
    exit 1
fi
# Replace only this script's previous sanitized source staging directory.
rm -rf .build/upstream
mkdir -p .build/upstream
git -C "$RFD4_SOURCE" archive "$RFD4_COMMIT" | tar -x -C .build/upstream
cp cloudbuild.yaml .build/
mkdir -p .validation
source_id=$(date -u +%Y%m%dT%H%M%SZ)
tar -czf .validation/source.tgz -C .build upstream cloudbuild.yaml
source_uri="gs://${BUCKET}/build-source/${source_id}.tgz"
gcloud storage cp .validation/source.tgz "$source_uri"
gcloud builds submit --no-source --project="$PROJECT_ID" --config=cloudbuild.yaml \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
    --substitutions="_IMAGE=$IMAGE,_SOURCE=$source_uri" \
    --async --format='value(id)' | tee .build/build-id
echo "Build submitted; follow it with: gcloud beta builds log $(cat .build/build-id) --stream --project=$PROJECT_ID"
