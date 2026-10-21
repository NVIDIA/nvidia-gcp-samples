# RFD4-Proteína deployment design

Clone the public [RosettaCommons/RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) repository at the commit pinned from [PR #52](https://github.com/RosettaCommons/RFD4-Proteina/pull/52). Cloud Build builds the root Dockerfile's `api` target. The FastAPI server, CPU tests, and portable Kubernetes manifests live upstream under `deploy/`; this sample contains only Google Cloud provisioning, build configuration, and the Autopilot overlay.

```text
Client → ClusterIP Service → GPU API Deployment → Regional persistent disk
                                     ↓
                            Resident model + queue
```

One A100 40 GB processes requests sequentially. The server uses the upstream inference configuration to load the pinned Hugging Face model and autoencoder once per startup. Requests return an ID for polling and artifact downloads. The 100 GiB volume stores the Hugging Face cache, request status, and generated CIFs; interrupted requests fail explicitly after restart.

GKE Autopilot provisions GPU nodes. `upstream/deploy/kubernetes/` contains the base manifests; `overlay/` selects A100, sets resource requests, and creates a regional `pd-balanced` StorageClass. One replica and `Recreate` updates avoid requiring two GPUs during deployment. The internal Service is accessed by port forwarding; public ingress requires authenticated HTTPS.

The build account has read access to the shared Cloud Build source bucket, write access to this sample's Artifact Registry repository, and log-writing access. Cleanup removes the workload, waits for disk reclamation, and deletes the cluster, registry, and network. Shared build staging, service accounts, and IAM grants are retained.
