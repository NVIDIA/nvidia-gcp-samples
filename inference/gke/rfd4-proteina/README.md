# RFD4-Proteína API on GKE

Deploy a persistent protein-design API on one NVIDIA A100 GPU on GKE Autopilot. The server loads the model once, queues requests, and saves checkpoints, status, and generated CIFs on a persistent disk.

## Prerequisites

- A billing-enabled Google Cloud project and quota for one A100 40 GB in your chosen region.
- `gcloud` (with the `beta` component installed for Cloud Logging build stream), `kubectl`, Git, and Python 3. Cloud Build builds the image; local Docker is optional.
- Permissions to create a VPC/subnet, GKE Autopilot cluster, Artifact Registry repository, builds, and service accounts (or set `BUILD_SA` and `NODE_SA` to existing pre-authorized service accounts).

Run commands from this sample directory. GPU pods remain billable until the workload or cluster is deleted.

## 1. Configure and prepare

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION=us-central1
gcloud config set project "$PROJECT_ID"
source scripts/env.sh
bash scripts/prepare.sh
```

This enables missing APIs, creates an Artifact Registry repository, and creates dedicated build/node service accounts (`${NAME}-build` and `${NAME}-node`) if they do not already exist. Resource names default to `rfd4-proteina`; override `NAME` before sourcing `scripts/env.sh` if needed.

## 2. Clone and build

```bash
git clone https://github.com/RosettaCommons/RFD4-Proteina.git upstream
git -C upstream fetch origin "$RFD4_COMMIT"
git -C upstream checkout --detach "$RFD4_COMMIT"
bash scripts/build.sh
gcloud beta builds log "$(cat .build/build-id)" --stream --project="$PROJECT_ID"
```

Confirm the build finishes with `SUCCESS`. Cloud Build uploads the `upstream` directory to the project's default Cloud Build staging bucket and builds the root Dockerfile's `api` target (CUDA 13.2 runtime and FastAPI server). Both image definitions and API code are available from source; no maintainer-built image is required. Builds take several minutes.

The pinned upstream commit contains the portable API and Kubernetes assets proposed in [RosettaCommons/RFD4-Proteina PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16). The pin is usable before merge; update it to a reviewed main-branch commit after merge. This recipe keeps only Google Cloud provisioning, Cloud Build configuration, and a GKE Autopilot Kustomize overlay (`overlay/kustomization.yaml`); see `upstream/deploy/README.md` for Docker, other clouds, and on-premises Kubernetes.

## 3. Create GKE Autopilot cluster and deploy

```bash
bash scripts/cluster.sh
bash scripts/deploy.sh
kubectl logs -n "$NAME" deployment/api --tail=30
```

This creates a dedicated VPC/subnet and a regional GKE Autopilot cluster. When the Deployment is applied, GKE Autopilot automatically provisions an A100 node with the latest available GPU driver (`cloud.google.com/gke-gpu-driver-version: latest`). The deployment script renders the upstream Kubernetes base with `overlay/kustomization.yaml` for A100 node selection, CUDA library paths, Autopilot-compliant Guaranteed QoS resources, `standard-rwo-regional` SSD storage (100 GiB regional `pd-balanced` PersistentVolumeClaim), namespace, and image digest. Startup downloads the roughly 8 GB checkpoint pair from upstream public HTTPS, caches it on disk, and loads the model. The Service is internal.

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
RFD4_SOURCE="$PWD/upstream" python3 upstream/deploy/api/smoke_test.py
```

To check persistence, restart the pod, then stop/restart port forwarding and retrieve a previous request's status and results:

```bash
kubectl rollout restart deployment/api -n "$NAME"
kubectl rollout status deployment/api -n "$NAME" --timeout=300s
```

## Limits and troubleshooting

One replica, one Uvicorn process, eight waiting requests, one to four replicates per request, uploads up to 4 MiB, and up to 512 transformed tokens. A restart preserves completed results and marks interrupted requests failed for resubmission. Checkpoints reload from disk after a pod restart. Delete old results periodically to stay within the 100 GiB volume. Public access requires authenticated HTTPS; this sample exposes no public endpoint.

The `standard-rwo-regional` disk is replicated across two zones in the region, allowing the Pod to reschedule into an alternate zone on restart if its original zone lacks GPU capacity. The running API reserves its GPU even when no requests are queued.

Generation uses the upstream medium preset without refolding or reward-guided search. The startup probe allows 30 minutes. Inspect logs/events if readiness fails:

```bash
kubectl get pods -n "$NAME" -o wide
kubectl describe pod -n "$NAME" -l app=rfd4-api
kubectl logs -n "$NAME" deployment/api --tail=100
```

## 5. Cleanup

Remove only this sample's named resources, including its results disk, cluster, image repository, and network, when finished:

```bash
bash scripts/cleanup.sh
```

See [PLAN.md](PLAN.md) for design references and [VALIDATION.md](VALIDATION.md) for the actual end-to-end result.
