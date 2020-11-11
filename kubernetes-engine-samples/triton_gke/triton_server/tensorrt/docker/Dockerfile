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

ARG trt_ngc_version=20.09

FROM nvcr.io/nvidia/tensorrt:${trt_ngc_version}-py3

LABEL maintainer="NVIDIA CORPORATION"

RUN apt-get update && apt-get install -y csh
RUN apt-get install -y tcsh

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    build-essential

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install required Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install onnx
RUN pip3 install pycuda
RUN pip3 install tensorflow-gpu==1.15.4

ENV PATH $PATH:/usr/src/tensorrt/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/src/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat

ADD TensorRT/demo/BERT /workspace/bert
ADD resnet /workspace/resnet

RUN apt-get update && apt-get install -y lsb-core && export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y gcsfuse && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get install -y apt-transport-https ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && mkdir /mnt/bert && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 30

WORKDIR /workspace/bert

ADD run.sh /workspace/bert

ENTRYPOINT ["bash", "run.sh"]


