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


# We modify the original Higgs data into 1GB chunks and append to different size for benchmarking. We also generate parquet files for our testing.

FROM nvcr.io/nvidia/rapidsai/rapidsai:0.16-cuda10.1-base-ubuntu18.04

RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rapids \
    && pip install -U gcsfs

ADD rapids.py /rapids
ADD entrypoint.sh /rapids

WORKDIR /rapids

ENTRYPOINT ["bash", "entrypoint.sh"]