# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export PROJECT_ID=<your_project_id>
export GCS_BUCKET=<your_project_id>/<your_name>
export REGION=us-central1
export ZONE=us-central1-f

python3 bert_squad2_qa_gpu.py             \
  --runner "DataflowRunner"               \
  --project "$PROJECT_ID"                 \
  --temp_location "gs://$GCS_BUCKET/tmp"  \
  --region "$REGION"                      \
  --worker_zone "$ZONE"                   \
  --worker_machine_type "n1-standard-4"   \
  --worker_harness_container_image "gcr.io/${PROJECT_ID}/tf-dataflow-<your_name>:latest"          \
  --experiment "worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver" \
  --experiment "use_runner_v2"                            \
  --subnetwork "regions/${REGION}/subnetworks/default"    \
  --autoscaling_algorithm=NONE                            \
  --num_workers=1                                         \
  --number_of_worker_harness_threads=4

