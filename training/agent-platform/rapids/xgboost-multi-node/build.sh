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


#!/usr/bin/env bash

docker build -t gcr.io/<account_name>/gcp_rapids_dask:latest .

docker push gcr.io/<account_name>/gcp_rapids_dask:latest
