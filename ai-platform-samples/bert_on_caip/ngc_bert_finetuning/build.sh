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

mkdir bert_for_tensorflow_v6 && cd bert_for_tensorflow_v6
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/bert_for_tensorflow/versions/20.06.4/zip -O bert_for_tensorflow_20.06.4.zip
unzip bert_for_tensorflow_20.06.4.zip
cd ..

docker build -t gcr.io/k80-exploration/tf_bert_gcsfuse:latest .

docker push gcr.io/k80-exploration/tf_bert_gcsfuse:latest
