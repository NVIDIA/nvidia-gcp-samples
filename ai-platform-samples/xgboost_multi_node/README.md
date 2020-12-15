# RAPIDS Dask on GCP with HIGGS Dataset
## Download the dataset
From the following https://archive.ics.uci.edu/ml/datasets/HIGGS download the entire dataset and move it to the  GCS bucket.

## Build the Docker container and push it to GCR
Change the build.sh to change the account name and then run
``` sh build.sh ```

## Customize the GCP AI Platform job
You can customize the GCP AI platform job using the rapids_distributed.yaml.

 * workerCount - the number of the workers that are needed for the job.
 * imageUri - is the image that is pushed to the GCR from the previous step.
 * masterType & workerType - is the the type of the machine that is needed for the job.
 * acceleratorConfig - specifies the type of GPU and the number of GPU's /machine that is requested for the job.
 * scaleTier - should be set CUSTOM to run custom containers in the GCP AI Platform

## Create a GCP AI Platform job
Cutomize the rapids_distributed.yaml to the account needs and then createa AI platform job.
``` gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --config rapids_distributed.yaml ```