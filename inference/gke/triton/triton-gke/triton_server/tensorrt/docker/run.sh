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

# gcp_bucket=${1:-"dlvm-dataset"}
# gcp_base_folder=${2:-"bert"}
# mkdir -p /mnt/bert
# gcsfuse --only-dir ${gcp_base_folder} \
#         --implicit-dirs ${gcp_bucket} /mnt/bert

engine_dir=${1:-"/mnt/bert/trt_engine"}
checkpoint_dir=${2:-"/workspace/bert/checkpoint"}
squad_dir=${3:-"/workspace/TensorRT/demo/BERT/squad"}
seq_length=${4:-"384"}
bert_model=${5:-"large"}

mkdir -p /results
bash scripts/download_squad.sh v2_0
mkdir -p ${engine_dir}
mkdir -p ${squad_dir}
mkdir -p ${checkpoint_dir} && cd ${checkpoint_dir}
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_qa_squad2_amp_384/versions/19.03.1/zip -O bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip
unzip bert_tf_ckpt_large_qa_squad2_amp_384_19.03.1.zip
cd /workspace/bert

python3 builder.py -m $checkpoint_dir/model.ckpt \
                   -o $engine_dir/bert_${bert_model}_${seq_length}_int8.engine \
                   -c $checkpoint_dir -v $checkpoint_dir/vocab.txt \
                   --squad-json $squad_dir/dev-v2.0.json -b 16 \
                   -s $seq_length --fp16 --int8 --strict -imh -iln

cd /workspace/resnet
tar -xzvf resnet50v1.onnx.tar.gz

python3 onnx_to_tensorrt.py --explicit-batch \
                      --onnx resnetv1.onnx \
                      --fp16 \
                      --int8 \
                      --calibration-cache="cache/resnet50.cache" \
                      -o $engine_dir/resnet50_8-16-32_int8.engine