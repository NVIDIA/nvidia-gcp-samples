#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
gcloud container clusters get-credentials "$NAME" --zone="$ZONE" --project="$PROJECT_ID"
digest=$(gcloud artifacts docker images describe "$IMAGE" --project="$PROJECT_ID" --format='value(image_summary.digest)')
export DEPLOY_IMAGE="${IMAGE%:*}@${digest}"
mkdir -p .build
python3 - <<'PYOVERLAY'
import json
import os
from pathlib import Path
# Generate the provider overlay over the upstream portable base.
overlay = {
    "apiVersion": "kustomize.config.k8s.io/v1beta1", "kind": "Kustomization",
    "namespace": os.environ["NAME"],
    "resources": ["../upstream/deploy/kubernetes"],
    "images": [{"name": "rfd4-api", "newName": os.environ["DEPLOY_IMAGE"].split("@")[0],
                "digest": os.environ["DEPLOY_IMAGE"].split("@")[1]}],
    "patches": [
        {"target": {"kind": "Namespace", "name": "rfd4-proteina"},
         "patch": json.dumps([{"op": "replace", "path": "/metadata/name", "value": os.environ["NAME"]}])},
        {"target": {"kind": "PersistentVolumeClaim", "name": "data"},
         "patch": json.dumps([{"op": "add", "path": "/spec/storageClassName", "value": "standard-rwo"}])},
        {"target": {"kind": "Deployment", "name": "api"},
         "patch": json.dumps([{"op": "add", "path": "/spec/template/spec/nodeSelector",
                              "value": {"cloud.google.com/gke-accelerator": "nvidia-tesla-a100"}}])},
    ],
}
Path(".build/overlay").mkdir(parents=True, exist_ok=True)
Path(".build/overlay/kustomization.yaml").write_text(json.dumps(overlay))
PYOVERLAY
kubectl kustomize .build/overlay > .build/api.yaml
kubectl apply -f .build/api.yaml
kubectl rollout status deployment/api -n "$NAME" --timeout=1800s
