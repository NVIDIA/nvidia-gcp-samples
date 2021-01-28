# TensorRT Bert Q&A Inference in GCP Dataflow

## Sample Overview

This is sample that deployed a optimized BERT large model with sequence-length 384, fine-tuned with squad Q&A task, then run inference with GCP dataflow. To run the sample, please first create a docker image and push to GCR registry with [build_and_push.sh](build_and_push.sh). Please make sure your git lfs is enabled so that you can download the engine file. In the sample code, we generate replicated documentation and questions to send, each request has a default batch size of 16 which match the optimial batch size of the TensorRT Engine profile. 

Once image is pushed successfully, make sure your local python environment has version 3.6 and properly installed with dataflow prerequsite, then dataflow job could be launch with:
```
export PROJECT_ID=[your gcp project ID]
export GCS_BUCKET=[your gcs bucket for temp location]
export REGION=us-central1
export ZONE=us-central1-a

python3 bert_squad2_qa_trt.py             \
  --runner "DataflowRunner"               \
  --project "$PROJECT_ID"                 \
  --temp_location "gs://$GCS_BUCKET/tmp"  \
  --region "$REGION"                      \
  --worker_zone "$ZONE"                   \
  --worker_machine_type "n1-standard-4"   \
  --worker_harness_container_image "gcr.io/${PROJECT_ID}/trt-dataflow:latest"          \
  --experiment "worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver" \
  --experiment "use_runner_v2"                            \
  --subnetwork "regions/${REGION}/subnetworks/default"    \
  --autoscaling_algorithm=NONE                            \
  --num_workers=1                                         \
  --number_of_worker_harness_threads=4  
```

We expect ~100 GPS(Query per second) on a T4 n1-standard-4 instance. Which calculates to 650k inference per dollar on a 345 million parameters model. This could be further optimized with analysis of beam performance and TensorRT API usage. To understand the sample, we recommend go through following materials:

**Documentation**
- [Machine learning patterns with Apache Beam and the Dataflow Runner](https://cloud.google.com/blog/products/data-analytics/ml-inference-in-dataflow-pipelines)
- [GPU User Guide in Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/guides/using-gpus)
- [TensorFlow GPU Dataflow Sample](https://cloud.google.com/dataflow/docs/samples/satellite-images-gpus)
- [Bert Optimization with TensorRT](https://github.com/NVIDIA/TensorRT/tree/master/demo/BERT)
- [Polygraphy - Simplified TensorRT Python API](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)
