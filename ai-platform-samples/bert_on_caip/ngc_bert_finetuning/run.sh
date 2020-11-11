#!/usr/bin/env bash
# Copyright 2020 NVIDIA Corporation
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

mkdir -p /mnt/bert
mkdir -p /results
gcsfuse --only-dir bert \
        --implicit-dirs dlvm-dataset /mnt/bert

batch_size=${1:-"3"}
learning_rate=${2:-"5e-6"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
num_gpu=${5:-"8"}
seq_length=${6:-"384"}
doc_stride=${7:-"128"}
bert_model=${8:-"large"}
squad_version=${9:-"1.1"}
epochs=${10:-"2.0"}

BASE_DIR=/mnt/bert
BERT_DIR=$BASE_DIR/checkpoint/bert_tf_v1_1_large_fp16_384_v2
DATA_DIR=$BASE_DIR/squad/v1.1
RESULT_DIR=$BASE_DIR/output
printf "Saving checkpoints to %s\n" "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5

mpirun -np $num_gpu \
    --allow-run-as-root -bind-to socket \
    python run_squad.py --vocab_file=$BERT_DIR/vocab.txt \
     --bert_config_file=$BERT_DIR/bert_config.json \
     --init_checkpoint=$BERT_DIR/model.ckpt-5474 \
     --output_dir=$RESULT_DIR --train_batch_size=$batch_size \
     --do_predict=True --predict_file=$DATA_DIR/dev-v1.1.json \
     --eval_script=$DATA_DIR/evaluate-v1.1.py \
     --do_train=True --train_file=$DATA_DIR/train-v1.1.json \
     --learning_rate=$learning_rate \
     --num_train_epochs=$epochs \
     --max_seq_length=$seq_length \
     --doc_stride=$doc_stride \
     --save_checkpoints_steps 1000 \
     --horovod --amp --use_xla
