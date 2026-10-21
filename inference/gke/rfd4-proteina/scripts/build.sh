#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RFD4_SOURCE="${RFD4_SOURCE:-$PWD/upstream}"
if ! git -C "$RFD4_SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
    git clone "$RFD4_REPO" "$RFD4_SOURCE"
fi
if ! git -C "$RFD4_SOURCE" cat-file -e "${RFD4_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$RFD4_SOURCE" fetch "$RFD4_REPO" "$RFD4_COMMIT"
fi
if ! git -C "$RFD4_SOURCE" cat-file -e "${RFD4_COMMIT}:Dockerfile"; then
    echo "The selected upstream commit must include its public Dockerfile." >&2
    exit 1
fi
# Build only committed upstream files and the sample's serving layer.
rm -rf .build/context
mkdir -p .build/context/upstream
git -C "$RFD4_SOURCE" archive "$RFD4_COMMIT" | tar -x -C .build/context/upstream
cp Dockerfile.api .build/context/
mkdir -p .build/context/api
cp -R api/app api/examples .build/context/api/
cp api/requirements.txt .build/context/api/
find .build/context/api -type d -name __pycache__ -prune -exec rm -rf {} +
gcloud builds submit .build/context \
    --project="$PROJECT_ID" \
    --config=cloudbuild.yaml \
    --gcs-source-staging-dir="gs://${PROJECT_ID}_cloudbuild/source" \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
    --substitutions="_IMAGE=$IMAGE" \
    --async --format='value(id)' | tee .build/build-id
echo "Build submitted; follow it with: gcloud beta builds log $(cat .build/build-id) --stream --project=$PROJECT_ID"
