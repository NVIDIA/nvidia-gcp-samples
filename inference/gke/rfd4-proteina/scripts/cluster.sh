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
if ! gcloud container clusters describe "$NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud container clusters create "$NAME" --zone="$ZONE" --project="$PROJECT_ID" \
        --release-channel=regular --machine-type=e2-standard-4 --num-nodes=1 \
        --network="$NAME" --subnetwork="$NAME" \
        --enable-ip-alias --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --service-account="$NODE_SA" --scopes=cloud-platform \
        --disk-size=100 --labels=sample=rfd4-proteina
fi
if ! gcloud container node-pools describe "$GPU_POOL" --cluster="$NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud container node-pools create "$GPU_POOL" --cluster="$NAME" --zone="$ZONE" --project="$PROJECT_ID" \
        --node-locations="$GPU_ZONES" \
        --machine-type=a2-highgpu-1g --accelerator=type=nvidia-tesla-a100,count=1,gpu-driver-version=latest \
        --image-type=COS_CONTAINERD --num-nodes=0 --disk-size=200 \
        --enable-autoscaling --total-min-nodes=0 --total-max-nodes=1 --location-policy=ANY \
        --service-account="$NODE_SA" --scopes=cloud-platform \
        --workload-metadata=GKE_METADATA --node-taints=nvidia.com/gpu=present:NoSchedule
fi
gcloud container clusters get-credentials "$NAME" --zone="$ZONE" --project="$PROJECT_ID"
