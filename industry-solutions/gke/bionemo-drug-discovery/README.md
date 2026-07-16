# BioNeMo Drug Discovery on GKE

This recipe deploys an AI-accelerated life sciences drug discovery dashboard on Google Kubernetes Engine (GKE) with NVIDIA GPUs.

The workflow analyzes protein drug targets, predicts structures, detects binding pockets, screens candidate compounds, generates novel molecules, predicts resistance mutations, and explains results with Gemini.

## What You Deploy

- An autoscaling GPU node pool using NVIDIA RTX PRO 6000 GPUs.
- NVIDIA device plugin for GPU scheduling.
- Hyperdisk Balanced persistent storage for model weights.
- A GenMol NIM service deployed inside the cluster.
- A Gradio dashboard that uses ESM-2, ESMFold, GenMol NIM, DiffDock NIM, RDKit, and Gemini.

## Prerequisites

- A Google Cloud project with billing enabled.
- Permissions to create GKE clusters, node pools, disks, LoadBalancer services, and IAM-scoped resources.
- GPU quota for `g4-standard-48` with `nvidia-rtx-pro-6000-vws` in the zone you choose.
- `gcloud`, `kubectl`, and `helm` installed locally or available in Cloud Shell.
- An NGC API key for pulling NVIDIA containers from `nvcr.io`.
- An NVIDIA API key for hosted DiffDock NIM calls.
- A Gemini API key for text and voice explanations.
- An existing VPC and subnet in the selected region.

## Files

- `app/app.py`: Gradio dashboard application for the drug discovery workflow.
- `manifests/storageclass.yaml`: Hyperdisk Balanced storage class.
- `manifests/pvc.yaml`: Persistent volume claim for model weights.
- `manifests/model-download-job.yaml`: One-time job that downloads ESM-2 and ESMFold into the PVC.
- `manifests/genmol-nim.yaml`: GenMol NIM deployment and internal service.
- `manifests/dashboard.yaml`: Dashboard deployment and external LoadBalancer service.

## Configure Environment

Open Cloud Shell or a terminal authenticated to Google Cloud.

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="europe-west8"
export ZONE="europe-west8-b"
export VPC_NAME="default"
export SUBNET_NAME="default"

export CLUSTER_NAME="bionemo-drug-discovery"
export NAMESPACE="bionemo-nim"

export NGC_API_KEY="<YOUR_NGC_API_KEY>"
export NVIDIA_API_KEY="<YOUR_NVIDIA_API_KEY>"
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
```

Set the active project and enable required APIs.

```bash
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com
```

## Create the GKE Cluster

Create the GKE cluster before adding the GPU node pool.

```bash
gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --network="${VPC_NAME}" \
  --subnetwork="${SUBNET_NAME}" \
  --enable-ip-alias \
  --machine-type="e2-standard-4" \
  --num-nodes=1 \
  --node-locations="${ZONE}" \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --enable-image-streaming \
  --release-channel="rapid" \
  --logging=SYSTEM,WORKLOAD \
  --monitoring=SYSTEM
```

Get cluster credentials.

```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}"
kubectl get nodes -o wide
```

## Add the GPU Node Pool

Create an autoscaling node pool for RTX PRO 6000 GPU workloads.

```bash
gcloud container node-pools create gpu-pool \
  --cluster="${CLUSTER_NAME}" \
  --zone="${ZONE}" \
  --machine-type="g4-standard-48" \
  --accelerator=type=nvidia-rtx-pro-6000-vws,count=1 \
  --num-nodes=1 \
  --min-nodes=1 \
  --max-nodes=4 \
  --enable-autoscaling \
  --node-locations="${ZONE}" \
  --disk-type="hyperdisk-balanced" \
  --disk-size="500GB" \
  --node-taints="nvidia.com/gpu=present:NoSchedule" \
  --metadata="install-nvidia-driver=true" \
  --scopes="https://www.googleapis.com/auth/cloud-platform"
```

Install the NVIDIA device plugin and verify that the GPU is schedulable.

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=300s
kubectl get nodes -o=custom-columns='NODE:.metadata.name,GPUs:.status.capacity.nvidia\.com/gpu'
```

## Create Storage, Namespace, and Secrets

Run these commands from this recipe directory.

```bash
kubectl apply -f manifests/storageclass.yaml

kubectl create namespace "${NAMESPACE}"
kubectl apply -f manifests/pvc.yaml

kubectl create secret docker-registry ngc-secret \
  --namespace="${NAMESPACE}" \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  --docker-email="unused@example.com"

kubectl create secret generic api-keys \
  --namespace="${NAMESPACE}" \
  --from-literal=gemini-api-key="${GEMINI_API_KEY}" \
  --from-literal=ngc-api-key="${NGC_API_KEY}" \
  --from-literal=nvidia-api-key="${NVIDIA_API_KEY}"
```

## Download ESM Models

Pre-download ESM-2 and ESMFold into the persistent volume. This avoids a long cold start when the dashboard launches.

```bash
kubectl apply -f manifests/model-download-job.yaml
kubectl wait --for=condition=complete job/model-download \
  --namespace="${NAMESPACE}" \
  --timeout=1200s
kubectl logs job/model-download --namespace="${NAMESPACE}"
```

## Deploy GenMol NIM

Deploy GenMol as an internal Kubernetes service.

```bash
kubectl apply -f manifests/genmol-nim.yaml
kubectl get pods --namespace="${NAMESPACE}" -l app=genmol-nim
```

GenMol NIM may need several minutes to initialize. On RTX PRO 6000 Blackwell GPUs, some GenMol container builds can report CUDA compatibility errors. The dashboard is configured to use the hosted NVIDIA API for GenMol and DiffDock calls when available, with the in-cluster GenMol endpoint available at `http://genmol-nim.bionemo-nim:8000`.

## Deploy the Dashboard

Create a ConfigMap from the local dashboard code, then deploy the dashboard.

```bash
kubectl create configmap dashboard-app-code \
  --namespace="${NAMESPACE}" \
  --from-file=app.py=app/app.py \
  --dry-run=client \
  -o yaml | kubectl apply -f -

kubectl apply -f manifests/dashboard.yaml
kubectl rollout status deployment/dashboard --namespace="${NAMESPACE}" --timeout=600s
```

Get the dashboard URL.

```bash
kubectl get service dashboard-service \
  --namespace="${NAMESPACE}" \
  -o jsonpath='http://{.status.loadBalancer.ingress[0].ip}{"\n"}'
```

If the output is empty, wait a few minutes and run the command again.

## Explore the Dashboard

Open the dashboard URL in your browser. The application includes these tabs:

- Target Discovery: choose a disease target such as EGFR, HIV-1 protease, BACE1, or CDK2.
- Target Analysis: use ESM-2 to inspect protein sequence properties and embeddings.
- Structure Prediction: use ESMFold to generate a 3D protein structure.
- Binding Sites: detect likely binding pockets on the predicted structure.
- Drug Screening: screen known drugs with RDKit and DiffDock NIM, then generate candidates with GenMol NIM.
- Resistance Analysis: estimate resistance mutation risk with ESM-2 masked language modeling.
- Target Comparison: compare protein targets side by side.
- GPU Benchmark: measure structure prediction performance.
- GPU Monitor: view current GPU utilization and memory use.

## Troubleshooting

Check all pods.

```bash
kubectl get pods --namespace="${NAMESPACE}"
```

Stream dashboard logs.

```bash
kubectl logs --namespace="${NAMESPACE}" -l app=dashboard -f --tail=50
```

Stream GenMol logs.

```bash
kubectl logs --namespace="${NAMESPACE}" -l app=genmol-nim -f --tail=50
```

Check PVC status.

```bash
kubectl get pvc --namespace="${NAMESPACE}"
```

Check GPU node capacity.

```bash
kubectl describe node -l cloud.google.com/gke-accelerator=nvidia-rtx-pro-6000-vws
```

Restart the dashboard after updating `app/app.py`.

```bash
kubectl create configmap dashboard-app-code \
  --namespace="${NAMESPACE}" \
  --from-file=app.py=app/app.py \
  --dry-run=client \
  -o yaml | kubectl apply -f -

kubectl rollout restart deployment/dashboard --namespace="${NAMESPACE}"
```

## Cleanup

Delete the GKE cluster. This removes the node pools, deployments, services, and pods.

```bash
gcloud container clusters delete "${CLUSTER_NAME}" \
  --zone="${ZONE}" \
  --quiet
```

Verify that no disks from the sample remain. Delete any orphaned Hyperdisk Balanced disks associated with the PVC.

```bash
gcloud compute disks list \
  --filter="zone:${ZONE}" \
  --format="table(name,zone,sizeGb,type,status)"

gcloud compute disks delete "<DISK_NAME>" --zone="${ZONE}" --quiet
```

The main cost drivers are the RTX PRO 6000 GPU nodes, the system node, the LoadBalancer, and the Hyperdisk Balanced volume. Delete the cluster and any orphaned disks when you are done.
