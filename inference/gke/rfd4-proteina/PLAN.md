# RFD4-Proteína deployment design

Clone a pinned `RosettaCommons/RFD4-Proteina` commit and build its root Dockerfile's `api` target. Deploy **one persistent GPU API server** on GKE Autopilot using a Kubernetes Deployment and ClusterIP Service. Load the flow model and autoencoder once and reuse upstream preprocessing, generation, and structure export. A bounded queue serializes inference.

```text
Client → Service → GPU API Deployment → Persistent disk (standard-rwo-regional)
                       ↓
               Resident model + queue
```

The API accepts annotated CIF uploads, returns a request ID, and exposes status and artifact downloads. One A100 40 GB processes one design at a time. A 100 GiB regional SSD-backed PersistentVolumeClaim (`standard-rwo-regional`, `pd-balanced`) stores cached checkpoints, inputs, status, and outputs. Pod restarts preserve completed results and mark interrupted requests failed. Public checkpoints are downloaded at startup; the API needs no Google Cloud credentials.

Keep portable serving code, the Dockerfile, and base manifests upstream under `deploy/`. This sample contains Google Cloud provisioning, Cloud Build configuration, and a static GKE Autopilot Kustomize overlay (`overlay/kustomization.yaml`). The README explicitly clones upstream and builds its API image with Cloud Build using the project's default Cloud Build staging bucket; Artifact Registry stores the resulting image. Dedicated, least-privilege build/node service accounts are created by default (or existing accounts can be configured via `BUILD_SA` and `NODE_SA`). The upstream serving PR supplies the reusable image and deployment assets.

The README creates the VPC/subnet, GKE Autopilot cluster, storage, and Service; demonstrates port-forwarded API calls; and documents cleanup. Deployment uses one replica and `Recreate` updates to avoid requiring a second GPU. Refolding, guided search, multiple replicas, and authenticated public ingress are later extensions.

Patterns reviewed: [NIM GKE setup](../nim/llm-nim/gcloud/README.md), [BioNeMo model storage and services](../../../industry-solutions/gke/bionemo-drug-discovery/README.md), and [Triton GPU deployment/probes](../triton/triton-gke/triton_server/triton-inference-server/templates/deployment.yaml). Consult current [GKE GPU guidance](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpus) for driver compatibility.
