# Setup
Please replace <your_project> with your project name along with <your_image> with an image name of your choice.
Below are the steps to run this project:
1. `bash build`
1. `export REGION=us-central1`
1. `export JOB_NAME=<a_very_intersting_job_name_with_data_and_time_possibly>`
1. adjust rapids_distributed.yaml file based on your preferences of the resources (e.g., 2 workers 4 cores each or 4 workers 8 cores each)
1. gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --config ./rapids_distributed.yaml
