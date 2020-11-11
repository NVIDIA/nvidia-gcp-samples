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

FROM nvcr.io/nvidia/tritonserver:20.09-py3-clientsdk

RUN apt update && apt install -y python3-pip --reinstall && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install tensorflow==1.15 jupyterlab && \
    apt update -y  && apt-get install -y systemd && \
    mkdir /workspace/notebooks

ADD client_src /workspace/notebooks/demo
ADD jupyterlab.service /etc/systemd/system/
ADD perf_client_queries /workspace/notebooks/

RUN systemctl enable jupyterlab.service

WORKDIR /workspace/notebooks


