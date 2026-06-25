# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nim/meta/llama3-8b-instruct:1.0.0
ENV TZ US/Pacific
ENV DEBIAN_FRONTEND noninteractive
ENV NFSDATA_FILESTORE_IPV4 10.237.234.194
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
USER root
RUN mkdir -p /home/nemo
COPY source/entrypoint_0.sh /home/nemo/entrypoint_0.sh
COPY source/http_respond_ready.py  /home/nemo/http_respond_ready.py
WORKDIR /
RUN mkdir -p /opt/nim/.cache
RUN chmod 777 /opt/nim/.cache
ENV NIM_CACHE_PATH=/opt/nim/.cache
WORKDIR /home/nemo
RUN apt-get update && apt-get install python3-pip -y
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
ENTRYPOINT ["/home/nemo/entrypoint_0.sh"]



