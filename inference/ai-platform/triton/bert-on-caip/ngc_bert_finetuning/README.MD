### AI Platform TF BERT Fine-tuning on GPU Guide

* Download the squad v1.1 dataset, in this example,
 we download to a GCS location `gs://dongm-gcp-shared/nv_dlexamples/bert/squad`, 
 modify run.sh accordingly
* User need to push the docker images to GCP container registory, 
here we used: `gcr.io/k80-exploration/tf_bert_gcsfuse:latest`, 
please modify build.sh accordingly to point to your own GCP projects. 
* To run this example, execute:
```
ngc registry model download-version nvidia/bert_tf_v1_1_large_fp16_384:2

export Project_ID=[your gcp project]
export IMAGE_URI=gcr.io/$Project_ID/tf_bert_gcsfuse:latest
export REGION=us-central1
export JOB_NAME=bert_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
    --master-image-uri $IMAGE_URI \
    --region $REGION \
    --master-accelerator count=8,type=nvidia-tesla-v100 \
    --master-machine-type n1-highmem-96 \
    --scale-tier custom
```