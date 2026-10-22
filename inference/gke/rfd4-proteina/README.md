# RFD4-Proteína API on GKE

Deploy a persistent protein-design API on one NVIDIA A100 GPU on GKE Autopilot. The server loads the model once, queues requests, and saves checkpoints, status, and generated CIFs on a persistent disk.

## Prerequisites

- A billing-enabled Google Cloud project and quota for one A100 40 GB in your chosen region.
- `gcloud` (with the `beta` component installed for Cloud Logging build stream), `kubectl`, Git, and Python 3. Cloud Build builds the image; local Docker is optional.
- Project policies must permit Cloud Build source staging (`serviceusage.services.use`) and Cloud DNS (`dns.googleapis.com`), which Autopilot uses during cluster creation.
- Permissions to enable APIs, create networking, GKE, storage, registry, build, and service-account resources, update project/bucket/repository IAM, and act as the build/node service accounts. Alternatively, have an administrator prepare existing accounts and set `BUILD_SA` and `NODE_SA` before sourcing the environment.

Run commands from this sample directory. GPU pods remain billable until the workload or cluster is deleted.

## 1. Configure and prepare

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION=us-central1
export STORAGE_ZONE_1=us-central1-b
export STORAGE_ZONE_2=us-central1-c
gcloud config set project "$PROJECT_ID"
source scripts/env.sh
bash scripts/prepare.sh
```

Choose two A100-capable zones in your region for disk replication. The pod must run in one of these zones.

This enables missing APIs, prepares the shared default Cloud Build source bucket and a dedicated Artifact Registry repository, and creates build/node service accounts (`${NAME}-build` and `${NAME}-node`) with scoped permissions. Resource names default to `rfd4-proteina`; override `NAME` before sourcing `scripts/env.sh` if needed. Use a dedicated name because the scripts reuse resources with that name.

## 2. Clone and build

```bash
git clone --filter=blob:none https://github.com/RosettaCommons/RFD4-Proteina.git upstream
git -C upstream fetch origin "$RFD4_COMMIT"
git -C upstream checkout --detach "$RFD4_COMMIT"
bash scripts/build.sh
gcloud beta builds log "$(cat .build/build-id)" --stream --project="$PROJECT_ID"
```

Confirm the build finishes with `SUCCESS`. Cloud Build builds the upstream root Dockerfile's `api` target from a clean archive of the pinned commit. The image uses the upstream default CUDA 13 profile.

The recipe pins `da2a917222ee332c321129323de040b276e6484a` from [RosettaCommons/RFD4-Proteina PR #52](https://github.com/RosettaCommons/RFD4-Proteina/pull/52), which adds the portable FastAPI server, API image target, and Kubernetes base to the public release. This commit can be built before merge; update the pin to a reviewed main-branch commit after merge. Model weights come from [nvidia/RFD4-Proteina on Hugging Face](https://huggingface.co/nvidia/RFD4-Proteina), using the revision selected in upstream `configs/inference.yaml`.

This sample contains Google Cloud provisioning, Cloud Build configuration, and the Autopilot overlay. See `upstream/deploy/README.md` for the portable Docker/Kubernetes guide.

## 3. Create GKE Autopilot cluster and deploy

```bash
bash scripts/cluster.sh
bash scripts/deploy.sh
kubectl logs -n "$NAME" deployment/api --tail=30
```

This creates a dedicated VPC/subnet and a regional GKE Autopilot cluster. GKE provisions an A100 node with its latest available driver. The overlay selects the GPU, sets resource requests, and creates a `rfd4-regional-rwo` StorageClass for a 100 GiB regional `pd-balanced` disk. The image is deployed by immutable digest.

On first startup, the server downloads the pinned Hugging Face model bundle to `/workspace/cache/huggingface` on the persistent disk. Later startups reuse the cache. The Service is internal.

## 4. Call the API

In one terminal:

```bash
kubectl port-forward -n "$NAME" service/api 8000:8000
```

In another terminal, from this sample directory:

```bash
curl -fsS http://localhost:8000/health/ready
curl -fsS -X POST http://localhost:8000/v1/designs \
  -F file=@upstream/deploy/api/examples/monomer-short.cif \
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

The input in `upstream/deploy/api/examples/` is a plain CIF version of the upstream short-monomer benchmark.

Interactive API documentation is at `http://localhost:8000/docs`. Or run the automated test to generate two designs, download CIFs, reject invalid input, and verify checkpoints loaded once:

```bash
python3 upstream/deploy/api/smoke_test.py
```

To check persistence, restart the pod, then stop/restart port forwarding and retrieve a previous request's status and results:

```bash
kubectl rollout restart deployment/api -n "$NAME"
kubectl rollout status deployment/api -n "$NAME" --timeout=1800s
```

## Limits and troubleshooting

One replica, one Uvicorn process, eight waiting requests, one to four replicates per request, uploads up to 4 MiB, and up to 512 transformed tokens. A restart preserves completed results and marks interrupted requests failed for resubmission. Checkpoints reload from disk after a pod restart. Delete old results periodically to stay within the 100 GiB volume. Public access requires authenticated HTTPS; this sample exposes no public endpoint.

The regional disk is replicated across two zones. Rescheduling requires GPU capacity in one of those two zones; it does not guarantee capacity throughout the region. The running API reserves its GPU even when no requests are queued.

Generation uses the upstream medium preset without refolding or reward-guided search. The startup probe allows 30 minutes. Inspect logs/events if readiness fails:

```bash
kubectl get pods -n "$NAME" -o wide
kubectl describe pod -n "$NAME" -l app=rfd4-api
kubectl logs -n "$NAME" deployment/api --tail=100
```

## 5. Cleanup

Remove this sample's workload, results disk, cluster, image repository, and network when finished:

```bash
bash scripts/cleanup.sh
```

Cleanup stops if disk reclamation times out; investigate the PVC/PV before retrying. The shared Cloud Build staging bucket and its source archives, service accounts, and IAM grants are retained. An administrator can remove sample-only accounts/grants and obsolete source archives when no longer needed. Do not delete a staging bucket used by other builds.

See [PLAN.md](PLAN.md) for design references and [VALIDATION.md](VALIDATION.md) for current validation coverage.
