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
 - In addition, user could leverage [Google Cloud shell](https://cloud.google.com/shell/docs/launching-cloud-shell) and further improve the user experience with [Boost Mode](https://cloud.google.com/shell/docs/how-cloud-shell-works#boost_mode)

## Samples

 - [Triton Autoscaling Example with TensorRT Optimization in Google Kubernetes Engine](kubernetes-engine-samples/triton_gke)
 - [BERT fine tuning, TensorRT optimization, Serve TensorRT Engine through Triton in AI Platform Prediction ](ai-platform-samples/bert_on_caip)
 - [XGBoost with LocalCUDACluster Dask Single Node Sample](ai-platform-samples/xgboost_single_node/gcsfs_localcuda)
 - [RAPIDS XGBoost hyperparameter optimization example](https://github.com/rapidsai/cloud-ml-examples/tree/main/gcp)
 - [RAPIDS/Spark on GCP Dataproc](https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-gcp.html)
 - [TensorRT Bert Q&A Inference in GCP Dataflow](dataflow-samples/bert-qa-trt-dataflow)

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

- Dong Meng (github: [mengdong](https://github.com/mengdong))
- Rajan Arora (github: [roarjn](https://github.com/roarjn))
- Ethem Can (github: [ethem-kinginthenorth](https://github.com/ethem-kinginthenorth))
- Arun Raman (github: [arunraman](https://github.com/arunraman))
