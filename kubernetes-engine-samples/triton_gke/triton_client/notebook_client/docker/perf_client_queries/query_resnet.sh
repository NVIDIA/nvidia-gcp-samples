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
SERVER_HOSTNAME=${2:-"localhost"} # need update public IP
precision=${3:-"int8"}
BATCH_SIZE=${4:-32}
MAX_LATENCY=${5:-50000}
MAX_CLIENT_THREADS=${6:-4}
MAX_CONCURRENCY=${7:-10}
MODEL_VERSION=${8:-1}
PERFCLIENT_PERCENTILE=${9:-90}
TIMESTAMP=$(date "+%y%m%d_%H%M")
OUTPUT_FILE_CSV="/results/perf_client/${MODEL_NAME}/results_${TIMESTAMP}.csv"

mkdir -p /results/perf_client/${MODEL_NAME}

# to run show multi-instance improvement
ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 2000 \
   --concurrency-range 1:${MAX_CONCURRENCY}:1 \\
   -v \
   -i gRPC \
   -u ${SERVER_HOSTNAME}:8001 \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -z \
   -f ${OUTPUT_FILE_CSV} \
   --percentile ${PERFCLIENT_PERCENTILE}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"

/workspace/install/bin/perf_client $ARGS
