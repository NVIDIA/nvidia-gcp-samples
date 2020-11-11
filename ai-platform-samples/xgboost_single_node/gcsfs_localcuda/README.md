# How To Run
Please replace <your_project> with your project name along with <your_image> with an image name of your choice.
Below are the steps to run this project:
1. `bash build.sh`
1. `export REGION=<pick_a_region>`
1. `export JOB_NAME=<a_very_intersting_job_name_with_data_and_time_possibly>`
1. adjust rapids_distributed.yaml file based on your preferences of the resources (e.g., 1 GPU ro 2 GPUs)
1. `gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --config ./rapids_distributed.yaml`

Please note that `count` and `--num-worker` should be the same value. 
Please also pay attention to pick a machine where it has at least (`--num-worker` x `--threads-per-worker` ) cores.
