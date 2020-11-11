### AI Platform TF BERT TRT optimization on GPU Guide

* User need to push the docker images to GCP container registory, 
here we used: `gcr.io/k80-exploration/bert_trt_gcsfuse:latest`, 
please modify build.sh accordingly to point to your own GCP projects. 
* To run this example, execute:
```
ngc registry model download-version nvidia/bert_tf_v1_1_large_fp16_384:2

export Project_ID=[your gcp project]
export IMAGE_URI=gcr.io/$Project_ID/bert_trt_gcsfuse:latest
export REGION=us-central1
export JOB_NAME=bert_trt_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
    --master-image-uri $IMAGE_URI  \
    --region $REGION \
    --master-accelerator count=1,type=nvidia-tesla-t4 \
    --master-machine-type n1-highmem-8 \
    --scale-tier custom
```