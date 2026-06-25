# NVIDIA GPU Accelerated Application Samples in Google Cloud Platform

**Table Of Contents**
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Samples](#samples)
- [Additional Resources](#additional-resources)
- [Known issues](#known-issues)
- [License](#license)
- [Maintainers](#maintainers)

## Description

This repository maintains sample applications designed for NVIDIA software tools integrated with Google Cloud Platform (GCP), e.g. AI platform, Dataproc, GKE, etc.

For select demonstrations, the sample code will be contained within this repository. For others, we will reference and link to exceptional demonstrations available outside of this repository.


## Prerequisites

 - [Install Google Cloud SDK on your laptop/client workstation](https://cloud.google.com/sdk/docs/install), so that `gcloud` SDK cli interface could be run on the client
 - In addition, user could leverage [Google Cloud shell](https://cloud.google.com/shell/docs/launching-cloud-shell)

## Samples

Samples are organized first by use case, then by Google Cloud service, then by NVIDIA library or stack.

### Agentic
 - Agent Platform / AIQ: [Nemotron AIQ example](agentic/agent-platform/aiq/nemotron-aiq-example)

### Inference
 - AI Platform / Triton: [BERT fine tuning, TensorRT optimization, and Triton serving in AI Platform Prediction](inference/ai-platform/triton/bert-on-caip)
 - Cloud Run / NIM: [LLM NIM on Google Cloud Run](inference/cloud-run/nim/llm-nim)
 - GKE / Dynamo: [Dynamo VLLM disaggregated deployment](inference/gke/dynamo/vllm-disaggregated)
 - GKE / Dynamo: [Dynamo G4 Kimi K2.5 reference deployments](inference/gke/dynamo/g4-kimi-k25)
 - GKE / NIM: [LLM NIM on Google Kubernetes Engine](inference/gke/nim/llm-nim)
 - GKE / TAO and Triton: [Building a computer vision service using NVIDIA NGC and Triton in Google Cloud](inference/gke/tao-triton/build-cv-service)
 - GKE / Triton: [Triton autoscaling example with TensorRT optimization in Google Kubernetes Engine](inference/gke/triton/triton-gke)
 - Agent Platform / NIM: [LLM NIM on Agent Platform Workbench](inference/agent-platform/nim/llm-nim/workbench)
 - Agent Platform / NIM: [LLM NIM on Agent Platform Colab Enterprise](inference/agent-platform/nim/llm-nim/colab-enterprise)
 - Agent Platform / NIM: [NeMo Retriever NIM on Agent Platform Workbench](inference/agent-platform/nim/nemo-retriever/workbench)
 - Agent Platform / Triton: [Triton in Agent Platform Prediction](inference/agent-platform/triton/prediction/triton_inference.ipynb)
 - Agent Platform / Triton: [XGBoost ensemble inference with Triton](inference/agent-platform/triton/prediction/xgboost_ensemble/simple_xgboost_example.ipynb)

### Training
 - AI Platform / RAPIDS: [XGBoost with LocalCUDACluster Dask single node sample](training/ai-platform/rapids/xgboost-single-node/gcsfs_localcuda)
 - AI Platform / RAPIDS: [Multi-node XGBoost training sample](training/ai-platform/rapids/xgboost-multi-node)
 - Agent Platform / NeMo RL: [NeMo RL GRPO quickstart](training/agent-platform/nemo-rl/vtc-nemo-rl/nemo_rl_grpo_quickstart.ipynb)

### Data Processing
 - BigQuery / RAPIDS: [BigQuery analytics with Dask on GPU](data-processing/bigquery/rapids/dask-bigquery-connector/bigquery_dataproc_dask_xgboost.ipynb)
 - Dataflow / TensorFlow: [BERT Q&A inference in Dataflow](data-processing/dataflow/tensorflow/bert-qa-tf-dataflow)
 - Dataflow / TensorFlow: [T5 inference in Dataflow](data-processing/dataflow/tensorflow/t5-dataflow-gpu-cpu)
 - Dataflow / TensorRT: [TensorRT BERT Q&A inference in Dataflow](data-processing/dataflow/tensorrt/bert-qa-trt-dataflow)

### Physical AI
 - Compute Engine / Isaac Lab and Newton: [Accelerate robot learning with NVIDIA Isaac Lab and Newton on Google Cloud](physical-ai/compute-engine/isaac-lab-newton)


## Additional Resources

See the following resources to learn more about NVIDIA NGC and GPU resources in Google Cloud Platform

**Documentation**

- [GPU in Google Cloud Platform](https://cloud.google.com/gpu)
- [Optimize GPU Performance in Google Cloud Platform](https://cloud.google.com/compute/docs/gpus/optimize-gpus)
- [Getting started with NGC on Google Cloud Platform](https://docs.nvidia.com/ngc/ngc-gcp-setup-guide/index.html#abstract)
- [DL Frameworks GPU Performance Optimization Recommendations](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html#broad-recs)
- [Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#abstract) for A100 GPU

## Known Issues

NA

## Contributions
Contributions are welcome. Developers can contribute by opening a [pull request](https://help.github.com/en/articles/about-pull-requests) and agreeing to the terms in [CONTRIBUTING.MD](CONTRIBUTING.MD).

## License

See [LICENSE](LICENSE).

## Maintainers

- Fortuna Zhang (github: [FortunaZhang](https://github.com/FortunaZhang))
- Dong Meng (github: [mengdong](https://github.com/mengdong))
- Rajan Arora (github: [roarjn](https://github.com/roarjn))
- Ethem Can (github: [ethem-kinginthenorth](https://github.com/ethem-kinginthenorth))
- Arun Raman (github: [arunraman](https://github.com/arunraman))
- Juan Pablo Guerra (github: [juanpabloguerra16](https://github.com/juanpabloguerra16))
