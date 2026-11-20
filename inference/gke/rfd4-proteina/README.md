# RFD4-Proteína API on GKE

Deploy RFD4-Proteína on GKE Autopilot with one NVIDIA A100 40 GB GPU and persistent storage.

## Prerequisites

- A billing-enabled Google Cloud project with A100 40 GB quota in your chosen region.
- An authenticated `gcloud` account with permissions to enable APIs, provision GKE, networking, registry, storage and service accounts, update their IAM policies, and act as the build/node accounts.
- `gcloud` with its `beta` component, `kubectl`, the [GKE authentication plugin](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin), Git, and Python 3 with `venv` and `pip`.
- Project policies that permit Cloud Build source staging (`serviceusage.services.use`) and Cloud DNS (`dns.googleapis.com`).

Use the same terminal for all steps. The GPU remains billable until the workload or cluster is deleted.

## 1. Configure

```bash
git clone https://github.com/NVIDIA/nvidia-gcp-samples.git
cd nvidia-gcp-samples/inference/gke/rfd4-proteina

export PROJECT_ID="YOUR_PROJECT_ID"
export REGION=us-central1
export STORAGE_ZONE_1=us-central1-b
export STORAGE_ZONE_2=us-central1-c

gcloud config set project "$PROJECT_ID"
source scripts/env.sh
bash scripts/prepare.sh
```

Choose two A100-capable zones in your region. Preparation checks the required commands, enables APIs, and creates the image repository and dedicated build/node service accounts.

## 2. Build the image

Clone the upstream model repository and build its GPU API image with Cloud Build:

```bash
git clone --filter=blob:none https://github.com/RosettaCommons/RFD4-Proteina.git upstream
git -C upstream fetch origin "$RFD4_COMMIT"
git -C upstream checkout --detach "$RFD4_COMMIT"
bash scripts/build.sh
gcloud beta builds log "$(cat .build/build-id)" --stream --project="$PROJECT_ID"
gcloud builds describe "$(cat .build/build-id)" --project="$PROJECT_ID" --format='value(status)'
```

Continue when the build status is `SUCCESS`.

## 3. Deploy

```bash
bash scripts/cluster.sh
bash scripts/deploy.sh
```

This creates the cluster and waits for the API to become ready. The first startup downloads model weights from [Hugging Face](https://huggingface.co/nvidia/RFD4-Proteina). Weights and results are stored on a 100 GiB regional persistent disk.

## 4. Generate a design

```bash
python3 -m venv .venv
.venv/bin/pip install zstandard
.venv/bin/python scripts/run_example.py
```

The example opens a local connection to the API, submits the upstream 100-residue benchmark, waits for completion, and downloads `results/<request-id>/design.cif`. It closes the connection when finished. Cold compilation can take 30 minutes or more; the example allows one hour (`--timeout` changes this).

To explore the API interactively, run the following command and open [Swagger UI](http://localhost:8000/docs). Press **Ctrl+C** to stop forwarding before continuing.

```bash
kubectl port-forward -n "$NAME" service/api 8000:8000
```

## 5. Cleanup

```bash
bash scripts/cleanup.sh
```

This permanently deletes the workload and its results disk, then the cluster, image repository, and network. The shared build staging bucket, source archives, service accounts, and IAM grants are retained. An administrator can remove sample-only resources after confirming they are unused.

## Customization and troubleshooting

- Set `NAME` before sourcing `scripts/env.sh` to use a dedicated resource name. The default is `rfd4-proteina`; existing resources with that name are reused.
- To use administrator-provided accounts, set `BUILD_SA` and `NODE_SA` before sourcing `scripts/env.sh`.
- If port 8000 is occupied, run the example with `--port 8001`.
- Keep one API replica and worker. Requests queue and run sequentially. Completed results survive a pod restart; interrupted requests must be resubmitted. Manage retention to stay within the disk capacity.
- The regional disk can attach in the two selected zones; GPU capacity must be available in one of them. API limits and supported tasks are reported by `/v1/info`; a token limit does not guarantee that a design fits GPU memory.
- Keep the service internal or add authenticated HTTPS before exposing it publicly. See the [upstream API guide](https://github.com/RosettaCommons/RFD4-Proteina/blob/659864b8f7cd322ba80c5602495916c146b20d8b/deploy/README.md) for its request contract and limits.

If deployment or readiness fails:

```bash
kubectl get pods -n "$NAME" -o wide
kubectl describe pod -n "$NAME" -l app=rfd4-api
kubectl logs -n "$NAME" deployment/api --tail=100
```

The startup probe covers weight loading; compilation occurs during inference. Example connection logs are in `.build/example-port-forward.log`.

If cleanup fails, resolve the reported error and rerun `bash scripts/cleanup.sh`. It checks for remaining volumes even if the namespace was already deleted, and waits for disk reclamation before deleting the cluster.
