# End-to-end validation

**PASS — GPU API workflow on GKE, 2026-10-14.** Built the pinned upstream
source through this recipe, deployed its image, generated designs, downloaded
artifacts, and verified persistence after a pod restart.

| Component | Tested value |
|---|---|
| Upstream source | `f3061df1166d51e7281fe7e637c6e6400fca4b83` ([PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16)) |
| Recipe revision | `2443cbb70edc4f649d9686dddce2cdbff49c9446` (before this report update) |
| GKE / location | `1.36.4-gke.1391002` / `us-central1`, GPU node in `us-central1-c` |
| GPU / driver | NVIDIA A100-SXM4 40 GB / `580.173.02` |
| Runtime | Python 3.12.3, PyTorch 2.14.1+cu132, CUDA 13.2 |
| Source build | `7fa12d1d-ea08-448b-86e8-0796842388f1` — SUCCESS |
| Image digest | `sha256:7bbadae03580aae00830e3c07147f78c69eff710dd44d3aba279ba0fd25ff63c` |

## Checks passed

- Followed README preparation, clone/pin, Cloud Build, cluster setup, deployment,
  port forwarding, API calls, smoke test, and restart commands.
- The image passed model imports, CLI help, dependency checks, and the build-time
  CPU `torch.compile` check. A fresh-cache CUDA compilation also passed inside
  the deployed image without installing packages or modifying its runtime.
- The real upstream smoke test passed: two designs on one resident model instance
  with `checkpoint_loads=1`, artifact downloads, malformed-input rejection,
  513-token rejection, and successful generation after the rejected request.
- Biotite parsed both generated CIFs; each has 100 residues and finite coordinates.
- After a Deployment restart, a new server instance became ready with one
  checkpoint load. Both new designs and two pre-upgrade artifacts downloaded
  byte-for-byte unchanged.
- Swagger UI rendered with six operations in Chromium; its screenshot was inspected.
- Four CPU-only API tests, shell/Python syntax, documentation links, manifest
  dry runs, and the assumption scan against tracked recipe sources passed.

| Seed | Duration | CIF bytes | Atoms / residues |
|---|---:|---:|---:|
| 42 | 25.8 s | 129,474 | 672 / 100 |
| 43 | 14.9 s | 119,602 | 597 / 100 |

These timings cover the included short monomer, not larger inputs or concurrency.

## Fix found during testing

The initial image loaded the model but failed real inference because Triton's
runtime helper could not find `Python.h`. PR #16 now installs `python3-dev`
and includes a CPU compilation check; this recipe pins that fix. The full build,
rollout, fresh-cache CUDA compilation, and API tests were repeated successfully
on the rebuilt image. A temporary header installation used for diagnosis was
confined to the replaced diagnostic pod.

## Scope and resources

The run reused the existing sample-owned cluster, GPU pool, registry, bucket,
100 GiB PVC, checkpoint cache, and completed results. VM and disk identities
were preserved; no IAM policies or unrelated resources were changed. Evidence,
logs, source hashes, and screenshots remain in the ignored `.validation/` directory.

Fresh cluster provisioning, cold checkpoint downloads, and cleanup were not
retested. Cleanup was intentionally skipped: the working deployment and its
billable resources remain running. Use the README cleanup command when finished.
