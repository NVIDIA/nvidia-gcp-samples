# RFD4-Proteína deployment design

Clone a pinned commit from the public [RosettaCommons/RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) repository. Cloud Build builds its root Dockerfile, then adds this sample's FastAPI layer with `Dockerfile.api`. The public repository currently provides a CLI container; the HTTP server and Kubernetes manifests live in this sample.

```text
Client → ClusterIP Service → GPU API Deployment → Regional persistent disk
                                     ↓
                            Resident model + queue
```

One A100 40 GB processes requests sequentially. The server uses the upstream inference configuration to load the pinned Hugging Face model and autoencoder once per startup. Requests return an ID for polling and artifact downloads. The 100 GiB volume stores the Hugging Face cache, request status, and generated CIFs; interrupted requests fail explicitly after restart.

GKE Autopilot provisions GPU nodes. `kubernetes/` contains the base manifests; `overlay/` selects A100, sets resource requests, and creates a regional `pd-balanced` StorageClass. One replica and `Recreate` updates avoid requiring two GPUs during deployment. The internal Service is accessed by port forwarding; public ingress requires authenticated HTTPS.

The build account has read access to the shared Cloud Build source bucket, write access to this sample's Artifact Registry repository, and log-writing access. Cleanup removes the workload, waits for disk reclamation, and deletes the cluster, registry, and network. Shared build staging, service accounts, and IAM grants are retained.
