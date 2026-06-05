# Dynamo on GCP g4 (RTX PRO 6000 Blackwell) — Kimi K2.5

NVIDIA Dynamo + SGLang reference deployments for **Kimi K2.5** on Google Cloud's `g4-standard-384` instance (8× RTX PRO 6000 Blackwell, SM_120). 2-node GKE topology, TP=8 + PP=2.

## Quantizations

| Quantization | Directory |
|---|---|
| **NVFP4** (recommended for Blackwell) | [`kimi-k25-nvfp4/`](kimi-k25-nvfp4/) |
| **Native INT4** | [`kimi-k25-int4/`](kimi-k25-int4/) |

Each subdirectory contains its own deployment YAMLs, benchmark scripts, and result tables.

## Why NVFP4 on this hardware

RTX PRO 6000 Blackwell (SM_120) has **native FP4 Tensor Cores**. NVFP4 uses these directly via FlashInfer CUTLASS NVFP4 GEMM + MoE kernels. INT4 on the same hardware runs via Compressed Tensors WNA16 Marlin, which dequantizes INT4 weights to higher precision in registers before the GEMM (no native INT4 Tensor Core path on SM_120). At identical workload, NVFP4 delivers higher throughput than INT4 on this hardware.

## Reference

Google's published Kimi K2.5 references:
- NVFP4: <https://github.com/shivajid/sglang-rtx-pro-6000/tree/main/models/KimiK2.5/nvfp4>
- INT4:  <https://github.com/shivajid/sglang-rtx-pro-6000/tree/main/models/KimiK2.5>
