# Validation

## Current public-source integration

Pinned source: `89cc8ef7c932f14cce4d141f44b598b2d322a7ec` from the public RosettaCommons/RFD4-Proteina repository. [Upstream PR #52](https://github.com/RosettaCommons/RFD4-Proteina/pull/52) adds the API Docker target, serving code, and portable Kubernetes manifests to the public release. The sample builds that target and overlays those manifests. The runner delegates Hugging Face loading and per-request configuration to upstream inference helpers.

Local checks passed on 2026-10-21:

- Four CPU-only HTTP API tests: resident worker serialization, persistence, error recovery, queue backpressure, startup health, and artifact-path restrictions.
- Ruff lint/format checks and shell syntax checks for all six scripts.
- Public upstream Hydra configuration composes with the wrapper's overrides; helper signatures accept the resident-model reuse call.
- The pinned Hugging Face model revision is public and ungated.
- Build/deploy scripts ran with mocked cloud commands: clean upstream source staging, upstream API build context, custom namespace, immutable image digest, regional StorageClass and its two replica zones were verified. These checks did not submit a build or contact a cluster.
- Source-only sample assumption scan and whitespace checks passed.

This revision requires a new container build and GPU inference run before it can be called end-to-end validated. Previous results below used different code, dependencies, and checkpoints. No GCP resources were provisioned for this update.

## Live E2E attempt, 2026-10-21

**BLOCKED by project policies before image build and GPU deployment.** A fresh recipe checkout cloned and pinned upstream PR #52 successfully. Preparation succeeded using the documented existing-service-account option; the caller cannot create service accounts or modify IAM in the test project.

- Cloud Build source staging failed with HTTP 403: an explicit IAM deny blocks `serviceusage.services.use` for the caller.
- Fresh GKE Autopilot provisioning entered `ERROR`: `constraints/gcp.restrictServiceUsage` disallows `dns.googleapis.com`, preventing creation of the cluster DNS zone. Cloud DNS was already enabled; enabling the API alone does not override this policy.
- Four CPU API tests, Ruff lint/format, and Kubernetes rendering passed against the fresh upstream checkout. No image build, GPU inference, artifact persistence, or live Swagger UI validation completed in this attempt.
- Cleanup completed: the test-owned cluster, network, subnet, and empty image repository were deleted. Follow-up inventories found no matching cluster, VM, disk, network, subnet, registry, firewall, or address resources. The pre-existing shared staging bucket and service account were preserved. The workload/PVC cleanup path was not exercised because deployment never reached that stage.

An administrator must resolve both policy restrictions before the documented E2E workflow can proceed. The recipe now explicitly lists Cloud DNS in API preparation and documents the policy prerequisites.

## Historical Autopilot validation

The following results were reported in [fork PR #1](https://github.com/juanpabloguerra16/nvidia-gcp-samples/pull/1), merged before this public-source update.

**PASS — GPU API workflow on GKE Autopilot, 2026-10-20.** Built the pinned
upstream source through this recipe using the default Cloud Build staging
bucket, deployed its image onto a regional GKE Autopilot cluster with a static
Kustomize overlay, generated designs, downloaded artifacts, and verified
persistence after a pod restart.

| Component | Tested value |
|---|---|
| Upstream source | `f3061df1166d51e7281fe7e637c6e6400fca4b83` ([PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16)) |
| GKE / location | `1.36.4-gke.1391002` (Autopilot) / `europe-west4` |
| GPU / driver | NVIDIA A100-SXM4 40 GB (`cloud.google.com/gke-gpu-driver-version: latest`) |
| Storage | 100 GiB `standard-rwo-regional` (`pd-balanced` regional PD) |
| Runtime | Python 3.12.3, PyTorch 2.14.1+cu132, CUDA 13.2 |
| Source build | `e1c5ad12-3975-4afc-bacf-4c4aa515af55` — SUCCESS |
| Image digest | `sha256:9a5685842a6d4b54df953adf304aa1f8199fa7326f87bb0872dbe29b820c2ca3` |

## Checks passed

- Followed README preparation, clone/pin, Cloud Build (`gcloud builds submit upstream`),
  GKE Autopilot cluster setup (`gcloud container clusters create-auto`), static
  Kustomize overlay deployment, port forwarding, API calls, smoke test, and
  restart commands.
- The upstream smoke test (`upstream/deploy/api/smoke_test.py`) passed: two
  designs on one resident model instance with `checkpoint_loads=1`, artifact
  downloads, malformed-input rejection, 513-token rejection, and successful
  generation after the rejected request.
- After a Deployment restart (`kubectl rollout restart deployment/api`), a new
  server instance became ready in ~100 seconds from cached checkpoints on the
  `standard-rwo-regional` volume (`checkpoint_loads=1`), and pre-restart
  artifacts remained intact.

| Seed | Duration | CIF bytes | Atoms / residues |
|---|---:|---:|---:|
| 42 | 32.38 s | 129,474 | 672 / 100 |
| 43 | 16.76 s | 119,602 | 597 / 100 |

These timings cover the included short monomer, not larger inputs or concurrency.
