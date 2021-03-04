# Copyright 2021 NVIDIA Corporation
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


from typing import Text, List, Any
import copy
import os as os
import sys as sys
import numpy as np
import apache_beam as beam
from apache_beam.utils import shared
from apache_beam import pvalue
import tensorflow as tf
import tensorflow_text
import logging
from apache_beam.utils import shared
from apache_beam.options.pipeline_options import PipelineOptions
import time


class MyModel():
    def __init__(self, model):
        self.model = model

class DoManualInference(beam.DoFn):
    def __init__(self, shared_handle, saved_model_path):
        self._shared_handle = shared_handle
        self._saved_model_path = saved_model_path
  
    def setup(self):
       # setup is a good place to initialize transient in-memory resources.
        def initialize_model():
            import tensorflow as tf
            import os
            src='/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11'
            dst='/usr/local/lib/python3.6/dist-packages/tensorflow/python/libcusolver.so.10'
            try: 
                os.symlink(src, dst)
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                pass
            # Load a potentially large model in memory. Executed once per process.
            return MyModel(tf.saved_model.load(self._saved_model_path, ["serve"]))
      
        self._model = self._shared_handle.acquire(initialize_model)

    def process(self, element: List[Text]) -> List[Any]:
        yield (self.predict(element))
                  
    def predict(self, inputs: List[Text]) -> List[Any]:
        batch = list(inputs)
        batch_size = len(batch)
        bs = 16
        if batch_size < bs:
            # Pad the input batch to 10 elements to match the model expected input.
            pad = [''] * (bs - batch_size)
            batch.extend(pad)
        inference = self._model.model.signatures['serving_default'](tf.constant(batch))['outputs'].numpy()
        return inference[0:batch_size]



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    pipeline_options = PipelineOptions(save_main_session=True)
    questions = ["nq question: who is the ceo of nvidia", 
            "nq question: what is the population of the north varolina state",
            "nq question: what is the capital city of turkey",
            "nq question: when do babies start teeting"] * 4000
      
    bs = 16
    saved_model_path = 'model_b'+str(bs)
    start_time = time.time()
    with beam.Pipeline(options=pipeline_options) as p:
        shared_handle = shared.Shared()
        _ = (p | beam.Create(questions)
          | beam.BatchElements(min_batch_size=bs, max_batch_size=bs)
          | beam.ParDo(DoManualInference(shared_handle=shared_handle, saved_model_path=saved_model_path))
          | beam.Map(print))

