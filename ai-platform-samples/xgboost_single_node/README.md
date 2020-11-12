
This example showcases performance and TCO benefits of XGBoost model training with dask-cudf on GPUs (such as T4s) on GCP. 
[gcsfs_local_cpu](https://github.com/NVIDIA/nvidia-gcp-samples/tree/master/ai-platform-samples/xgboost_single_node/gcsfs_local_cpu) folder has the CPU XGBoost training example code and [gcsfs_localcuda](https://github.com/NVIDIA/nvidia-gcp-samples/tree/master/ai-platform-samples/xgboost_single_node/gcsfs_localcuda) folder has the single-node multi-GPU XGBoost training code. On GCP you can have upto 4x T4s, or 8x V100s or 16x A100s in a single node.
