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

This repository contains sample applications for NVIDIA software tool working with GCP platforms (ai platform, Dataproc, GKE, etc).
For some demos the sample code will be part of this repo, and for some demos we would link to some great demos outside of this repo.


## Prerequisites

 - [Install Google Cloud SDK on your laptop/client workstation](https://cloud.google.com/sdk/docs/install), so that `gcloud` SDK cli interface could be run on the client
 - In addition, user could leverage [Google Cloud shell](https://cloud.google.com/shell/docs/launching-cloud-shell)

## Samples

### Deep Learning Inference:
 - [Triton autoscaling example with TensorRT optimization in Google Kubernetes Engine](kubernetes-engine-samples/triton_gke)
 - [BERT fine tuning, TensorRT optimization, Serve TensorRT engine through Triton in AI Platform Prediction ](ai-platform-samples/bert_on_caip)
 - [Triton Inference Server application in Google Kubernetes Engine](https://cloud.google.com/blog/products/compute/triton-inference-server-in-gke-nvidia-google-kubernetes)
 - [Triton GKE Marketplace application](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/triton-inference-server), [Blog](https://cloud.google.com/blog/products/compute/triton-inference-server-in-gke-nvidia-google-kubernetes)
 - [AlphaFold batch inference with Vertex AI Pipelines](https://github.com/GoogleCloudPlatform/vertex-ai-alphafold-inference-pipeline)
 - [Triton in Vertex AI Prediction](https://github.com/NVIDIA/nvidia-gcp-samples/blob/master/vertex-ai-samples/prediction/triton_inference.ipynb)

### Machine Learning and Data Science:
 - [XGBoost with LocalCUDACluster Dask single node sample](ai-platform-samples/xgboost_single_node/gcsfs_localcuda)
 - [RAPIDS XGBoost hyperparameter optimization example](https://github.com/rapidsai/cloud-ml-examples/tree/main/gcp)
 - [XGBoost ensemble inference with Triton](https://github.com/NVIDIA/nvidia-gcp-samples/blob/master/vertex-ai-samples/prediction/xgboost_ensemble/simple_xgboost_example.ipynb)

### Big Data Analytics:
 - [RAPIDS/Spark on GCP Dataproc](https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-gcp.html)
 - [Churn example with Spark RAPIDS on GCP Dataproc](https://github.com/GoogleCloudPlatform/datalake-modernization-workshops/tree/main/spark-rapids-churn)
 - [TensorRT intergration with Dataflow](https://github.com/apache/beam/blob/master/sdks/python/apache_beam/ml/inference/tensorrt_inference.py)
 - [TensorRT Bert Q&A inference in GCP Dataflow](dataflow-samples/bert-qa-trt-dataflow)
 - [BigQuery analytics with Dask on GPU](https://github.com/NVIDIA/nvidia-gcp-samples/blob/master/bigquery-samples/dask-bigquery-connector/bigquery_dataproc_dask_xgboost.ipynb)

### End to End Deep Learning:
 - [Building a Computer Vision Service Using NVIDIA NGC and Triton in Google Cloud](https://info.nvidia.com/ngc-google-cloud-computer-vision-webinar.html)
 - [NVIDIA Merlin recommender system on GCP Vertex AI](https://github.com/GoogleCloudPlatform/nvidia-merlin-on-vertex-ai)
 - [AutoML Videl Edge on NVIDIA GPU](https://github.com/google/automl-video-ondevice) 

### NIM Microservice and NeMo:
 - [LLM NIM on Google Kubernetes Engine](https://github.com/NVIDIA/nvidia-gcp-samples/tree/nim-gcp/nim-samples/llm-nim/gke)
 - [LLM NIM on GCP Vertex AI Workbench](https://github.com/NVIDIA/nvidia-gcp-samples/tree/nim-gcp/nim-samples/llm-nim/vertexai/workbench)
 - [LLM NIM on GCP Vertex AI Colab Enterprise](https://github.com/NVIDIA/nvidia-gcp-samples/tree/nim-gcp/nim-samples/llm-nim/vertexai/colab-enterprise)
 - [LLM NIM on GCP CloudRun](https://github.com/NVIDIA/nvidia-gcp-samples/tree/nim-gcp/nim-samples/llm-nim/cloudrun)
 - [NeMo Retriever NIM on GCP Vertex AI Workbench](https://github.com/FortunaZhang/nvidia-gcp-samples/tree/nim-gcp/nim-samples/nemo-retriever/vertexai/workbench)


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

## License

See [LICENSE](LICENSE).

## Maintainers

- Fortuna Zhang (github: [FortunaZhang](https://github.com/FortunaZhang))
- Dong Meng (github: [mengdong](https://github.com/mengdong))
- Rajan Arora (github: [roarjn](https://github.com/roarjn))
- Ethem Can (github: [ethem-kinginthenorth](https://github.com/ethem-kinginthenorth))
- Arun Raman (github: [arunraman](https://github.com/arunraman))
