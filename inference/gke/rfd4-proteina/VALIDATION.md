# End-to-end validation

**In progress — 2026-10-14.** Testing the PR recipe on GKE with one NVIDIA
A100-SXM4 40 GB, driver 580.173.02, and the upstream CUDA 13.2 runtime.

The recipe now pins `f3061df1166d51e7281fe7e637c6e6400fca4b83` from
[upstream PR #16](https://github.com/RosettaCommons/RFD4-Proteina/pull/16).

## Issue found and fixed

The first full source build and rollout succeeded, but real inference failed
when Triton compiled its CUDA runtime helper: `Python.h` was missing.
PR #16 now installs `python3-dev` and runs a CPU `torch.compile` build check.
The recipe pin includes that fix. A fresh source build is running; a full pass
has not yet been established for the fixed image.

Installing the missing headers temporarily in the diagnostic pod confirmed the
fix: two real designs completed, each with 100 residues and finite coordinates.
The upstream smoke test passed artifact downloads, malformed-input rejection,
513-token rejection, subsequent recovery, and one model load per server instance.
These diagnostic results do not substitute for testing the rebuilt image.

## Checks and scope

- Followed the README: authenticated, cloned/pinned upstream, prepared build
  resources, built through the recipe, and applied its GKE overlay.
- Four upstream CPU-only API tests, shell/Python syntax, documentation links,
  manifest server-side dry runs, and the tracked-recipe assumption scan passed.
- Reused the existing sample-owned cluster, GPU pool, registry, bucket, PVC,
  checkpoint cache, and completed results; no IAM policies were changed.
- Recorded hashes of pre-upgrade results for persistence checks after rollout.
- Cleanup is intentionally skipped to preserve the running sample and its data.
  Cold checkpoint download and fresh cluster provisioning are not retested.
