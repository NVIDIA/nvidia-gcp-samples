# RFD4-Proteína API on GKE

Deploy a persistent protein-design API on one NVIDIA A100 GPU. The server loads the model once, queues requests, and saves checkpoints, status, and generated CIFs on a persistent disk. Inference runs inside a **Deployment**, with one request at a time; no Kubernetes Jobs are needed.

## Prerequisites

- A billing-enabled Google Cloud project and quota for one A100 40 GB in your chosen region. GPU nodes must have a driver compatible with the upstream CUDA 13.2 runtime.
- `gcloud`, `kubectl`, Git, Python 3, and `rg`. Cloud Build builds the image; local Docker is optional.
- Permissions to create a VPC/subnet, GKE cluster, bucket, image repository, and builds; permission to enable missing APIs and act as build/node service accounts.
- Existing build/node service accounts. By default scripts use the project's Compute Engine default service account. The build account needs log-writing, staging-object reading, and Artifact Registry writing permissions; the node account needs registry reading and GKE logging/monitoring permissions. An administrator must grant missing roles. Set `BUILD_SA` and `NODE_SA` to use dedicated accounts. Scripts do not change IAM policies.
- Access to `RosettaCommons/RFD4-Proteina` and permission to use its code and model checkpoints under the upstream terms. Authenticate Git if the repository requires it.

Run commands from this sample directory. GPU nodes remain billable until cleanup.

## 1. Configure and prepare

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION=us-central1
export ZONE=us-central1-a
# Choose GPU-capable zones in the same region; the pool runs at most one node.
export GPU_ZONES=us-central1-b,us-central1-c
gcloud config set project "$PROJECT_ID"
source scripts/env.sh
bash scripts/prepare.sh
```

This creates a build-staging bucket and image repository. The API uses disk storage and needs no bucket credentials. Resource names default to `rfd4-proteina`; override `NAME` before sourcing the environment if needed. Choose a dedicated name because scripts reuse resources with that name.

## 2. Clone and build

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git upstream
git -C upstream fetch origin "$RFD4_COMMIT"
git -C upstream checkout --detach "$RFD4_COMMIT"
bash scripts/build.sh
gcloud beta builds log "$(cat .build/build-id)" --stream --project="$PROJECT_ID"
```

Confirm the build finishes with `SUCCESS`. Cloud Build builds the upstream root Dockerfile's `api` target (CUDA 13.2 runtime and FastAPI server). Both image definitions and API code are available from source; no maintainer-built image is required. Builds take several minutes.

The pinned upstream commit contains the portable API and Kubernetes assets proposed in [RosettaCommons/RFD4-Proteina PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16). The pin is usable before merge; update it to a reviewed main-branch commit after merge. This recipe keeps only Google Cloud provisioning, Cloud Build configuration, and a GKE overlay; see `upstream/deploy/README.md` for Docker, other clouds, and on-premises Kubernetes.

## 3. Create GKE and deploy

```bash
bash scripts/cluster.sh
bash scripts/deploy.sh
kubectl logs -n "$NAME" deployment/api --tail=30
```

This creates a dedicated VPC/subnet, a CPU node, and an autoscaling GPU pool that starts at zero and grows to at most one `a2-highgpu-1g` node when the API needs it. The `ANY` location policy considers available capacity across the configured zones. GKE installs its latest available GPU driver on COS. The deployment script renders the upstream Kubernetes base with a GKE overlay for A100 node selection, `standard-rwo` storage, namespace, and image digest. It uses a 100 GiB PersistentVolumeClaim. Startup downloads the roughly 8 GB checkpoint pair from upstream public HTTPS, caches it on disk, and loads the model. The Service is internal.

## 4. Call the API

In one terminal:

```bash
kubectl port-forward -n "$NAME" service/api 8000:8000
```

In another terminal, from this sample directory:

```bash
curl -fsS http://localhost:8000/health/ready
curl -fsS -X POST http://localhost:8000/v1/designs \
  -F file=@upstream/data/benchmarks/monomer/monomer-short.cif \
  -F num_replicates=1 -F seed=42
```

Copy the returned `id`, then poll status and list artifacts:

```bash
export REQUEST_ID="RETURNED_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID"
curl -fsS "http://localhost:8000/v1/designs/$REQUEST_ID/results"
```

When status is `succeeded`, download an artifact using its `download` path:

```bash
export DOWNLOAD_PATH="PATH_RETURNED_BY_RESULTS"
curl -fsS "http://localhost:8000$DOWNLOAD_PATH" -o design.cif
```

Interactive API documentation is at `http://localhost:8000/docs`. Or run the automated test to generate two designs, download CIFs, reject invalid input, and verify checkpoints loaded once:

```bash
python3 scripts/smoke_test.py
```

To check persistence, restart the pod, then stop/restart port forwarding and retrieve a previous request's status and results:

```bash
kubectl rollout restart deployment/api -n "$NAME"
kubectl rollout status deployment/api -n "$NAME" --timeout=300s
```

## Limits and troubleshooting

One replica, one Uvicorn process, eight waiting requests, one to four replicates per request, uploads up to 4 MiB, and up to 512 transformed tokens. A restart preserves completed results and marks interrupted requests failed for resubmission. Checkpoints reload from disk after a pod restart. Delete old results periodically to stay within the 100 GiB volume. Public access requires authenticated HTTPS; this sample exposes no public endpoint.

The disk is zonal: after provisioning, the pod needs GPU capacity in that disk's zone. The running API reserves its GPU even when no requests are queued.

Generation uses the upstream medium preset without refolding or reward-guided search. The startup probe allows 30 minutes. Inspect logs/events if readiness fails:

```bash
kubectl get pods -n "$NAME" -o wide
kubectl describe pod -n "$NAME" -l app=rfd4-api
kubectl logs -n "$NAME" deployment/api --tail=100
```

If cluster creation stalls with no registered CPU node, inspect that sample node's serial log. A transient bootstrap connection refusal to the control plane can require restarting the newly created node after the API is healthy:

```bash
gcloud compute instances list --project="$PROJECT_ID" --filter="name:gke-${NAME}" \
  --format='table(name,zone,status)'
gcloud compute instances reset SAMPLE_NODE_NAME --zone="$ZONE" --project="$PROJECT_ID"
```

## 5. Cleanup

Remove only this sample's named resources, including its results disk and build-staging bucket, when finished:

```bash
bash scripts/cleanup.sh
```

See [PLAN.md](PLAN.md) for design references and [VALIDATION.md](VALIDATION.md) for the actual end-to-end result.
