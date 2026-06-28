# Dynamo on GCP g4 (RTX PRO 6000 Blackwell)

NVIDIA Dynamo + SGLang reference deployments on Google Cloud's `g4-standard-384` instance (8× RTX PRO 6000 Blackwell, SM_120). 2-node GKE topology (TP=8 + PP=2).

## Models

| Model | Quantization | Directory |
|---|---|---|
| **Kimi K2.5** | NVFP4 (recommended for Blackwell) | [`kimi-k25-nvfp4/`](kimi-k25-nvfp4/) |
| **Kimi K2.5** | Native INT4 | [`kimi-k25-int4/`](kimi-k25-int4/) |
| **DeepSeek V4 Pro** | NVFP4 | [`ds-v4-pro-nvfp4/`](ds-v4-pro-nvfp4/) |

Each subdirectory contains its own deployment YAMLs, benchmark results, and README.

## Why NVFP4 on this hardware

RTX PRO 6000 Blackwell (SM_120) has **native FP4 Tensor Cores**. NVFP4 uses these directly via FlashInfer CUTLASS NVFP4 GEMM + MoE kernels. INT4 on the same hardware runs via Compressed Tensors WNA16 Marlin, which dequantizes INT4 weights to higher precision in registers before the GEMM (no native INT4 Tensor Core path on SM_120). At identical workload, NVFP4 delivers higher throughput than INT4 on this hardware.

## References

Google's published SGLang on RTX PRO 6000 references (multiple models): <https://github.com/shivajid/sglang-rtx-pro-6000/tree/main>
