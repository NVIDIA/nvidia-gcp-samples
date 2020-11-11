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

MODEL_NAME=${1:-"Bert-2"}
SERVER_HOSTNAME=${2:-"localhost"} # need update public IP
MODEL_VERSION=${8:-1}
precision=${3:-"int8"}
BATCH_SIZE=${4:-16}
MAX_LATENCY=${5:-50000}
MAX_CLIENT_THREADS=${6:-4}
MAX_CONCURRENCY=${7:-3}
SEQ_LENGTH=${9:-"384"}
PERFCLIENT_PERCENTILE=${10:-90}
MAX_TRIALS=${11:-40}
TIMESTAMP=$(date "+%y%m%d_%H%M")
OUTPUT_FILE_CSV="/results/perf_client/${MODEL_NAME}/results_${TIMESTAMP}.csv"

mkdir -p /results/perf_client/${MODEL_NAME}

ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 1000 \
   -r ${MAX_TRIALS} \
   -v \
   --concurrency-range 1:${MAX_CONCURRENCY}:1 \
   -i gRPC \
   -u ${SERVER_HOSTNAME}:8001 \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -z \
   -f ${OUTPUT_FILE_CSV} \
   --shape input_ids:${SEQ_LENGTH} \
   --shape segment_ids:${SEQ_LENGTH} \
   --shape input_mask:${SEQ_LENGTH} \
   --percentile ${PERFCLIENT_PERCENTILE}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"
/workspace/install/bin/perf_client $ARGS