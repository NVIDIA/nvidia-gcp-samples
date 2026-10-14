# Validation status

Validated on 2026-10-14 with an NVIDIA A100-SXM4 40 GB on GKE.

## Current upstream integration

The recipe clones `RosettaCommons/RFD4-Proteina` and pins commit
`f3061df1166d51e7281fe7e637c6e6400fca4b83` from
[upstream PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16).
Portable serving code and Kubernetes manifests live upstream. This recipe adds
Cloud Build, Google Cloud provisioning, and a GKE overlay.

- The upstream root Dockerfile's API target built and uploaded successfully with
  Python 3.12.3, PyTorch 2.14.1 / CUDA 13.2, and TMol 0.1.59+cu132torch2.14.
  Model imports, CLI help, and dependency compatibility checks passed.
- Four upstream CPU-only API tests passed, covering serialization, persistence,
  interrupted requests, failure recovery, input limits, startup health, queue
  backpressure, and storage path containment.
- The portable Kubernetes base passed server-side creation dry runs. The GKE
  overlay passed a server-side dry run against the existing sample deployment;
  custom namespace and immutable-image-digest rendering also passed.
- Shell syntax, Python syntax, changed-file lint/format checks, and the standalone
  sample assumption scan passed. Source archives exclude generated files and credentials.

The build snapshot precedes final upstream comments, type annotations, and test
additions; the Dockerfile and serving behavior are unchanged. The current recipe
has **not** been rerun end to end with the newer CUDA 13.2 image. Its startup,
GPU inference, driver compatibility, and cleanup still require validation before
claiming a full pass for the pinned revision. The upstream PR must remain
accessible; update the pin after it merges.

## Earlier serving implementation

The earlier PyTorch 2.13 / CUDA 13.0 serving implementation passed real GKE tests:

- Provisioned a dedicated network, GKE cluster, CPU node, and one A100 GPU node.
- Generated two 100-residue designs on one resident model instance, downloaded
  CIFs, and verified finite coordinates with Biotite.
- Rejected malformed input and a 513-token request, then successfully handled
  another valid request.
- Restarted the API and retrieved completed artifacts byte-for-byte unchanged.
- Rendered and visually inspected the Swagger UI in Chromium.

Those results support the serving design; they do not establish GPU compatibility
for the newer runtime. Cleanup was not executed: the earlier deployment remains
running. No unrelated resources or IAM policies were modified.
