#!/usr/bin/env bash
# Copyright 2019 NVIDIA Corporation
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

MODEL_NAME=${1:-"ResNet50-INT8-2"}
MODEL_VERSION=${2:-1}
precision=${3:-"int8"}
BATCH_SIZE=${4:-1}
MAX_LATENCY=${5:-500}
MAX_CLIENT_THREADS=${6:-8}
MAX_CONCURRENCY=${7:-32}
SERVER_HOSTNAME=${8:-"localhost"} # need update public IP
STABILITY_PERCENTAGE=${11:-0.01}
MAX_TRIALS=${12:-1000000}
PERFCLIENT_PERCENTILE=${10:-90}
TIMESTAMP=$(date "+%y%m%d_%H%M")

# to run show multi-instance improvement
ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 1000 \
   -t ${MAX_CONCURRENCY} \
   -s ${STABILITY_PERCENTAGE} \
   -r ${MAX_TRIALS} \
   -v \
   -i gRPC \
   -u ${SERVER_HOSTNAME}:8001 \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -z \
   --percentile ${PERFCLIENT_PERCENTILE}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"

/workspace/install/bin/perf_client $ARGS