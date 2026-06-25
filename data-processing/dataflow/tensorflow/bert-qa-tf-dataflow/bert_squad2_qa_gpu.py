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

from typing import List, Any, Text, Tuple
import time
import logging
import os
import apache_beam as beam
from apache_beam.utils import shared
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
from tensorflow.compat.v1.saved_model import tag_constants

def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class TfModel():
    def __init__(self, model, vocab_file="vocab.txt"):
        import helpers.tokenization as tokenization
        with tf.Graph().as_default() as graph:
            self.vocab_file = vocab_file
            self.tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)
            self.do_lower_case = True
            self.max_seq_length = 384
            self.doc_stride = 128
            self.max_query_length = 64
            self.verbose_logging = True
            self.version_2_with_negative = False
            self.n_best_size = 20
            self.max_answer_length = 30
            self.model = model

class DoManualInference(beam.DoFn):
    def __init__(self, shared_handle, engine_path, batch_size):
        import collections
        self._shared_handle = shared_handle
        self._engine_path = engine_path
        self._batch_size = batch_size
        self._NetworkOutput = collections.namedtuple(
            "NetworkOutput",
            ["start_logits", "end_logits", "feature_index"])


    def setup(self):
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
            return TfModel(tf.saved_model.load(self._engine_path, ["serve"]))

        self._model = self._shared_handle.acquire(initialize_model)


    def process(self, element: Tuple[Text, List[Text]]) -> List[Any]:
        yield (self.predict(element))

    def predict(self, inputs: Tuple[Text, List[Text]]) -> List[Any]:
        import helpers.data_processing as dp
        import numpy as np
        import collections
        import time

        def question_features(tokens, question):
            # Extract features from the paragraph and question
            return dp.convert_example_to_features(tokens, question,
                                                  self._model.tokenizer,
                                                  self._model.max_seq_length,
                                                  self._model.doc_stride,
                                                  self._model.max_query_length)

        features = []
        doc_tokens = dp.convert_doc_tokens(inputs[0])
        ques_list = inputs[1]

        batch_size = len(ques_list)
        if batch_size < self._batch_size:
            # Pad the input batch to batch_size to match the model expected input.
            pad = [ques_list[0]] * (self._batch_size - batch_size)
            ques_list.extend(pad)

        for question_text in ques_list:
            features.append(question_features(doc_tokens, question_text)[0])

        input_ids_batch = np.array([feature.input_ids for feature in features]).squeeze()
        segment_ids_batch = np.array([feature.segment_ids for feature in features]).squeeze()
        input_mask_batch = np.array([feature.input_mask for feature in features]).squeeze()
        start_time = int(time.time())
        uids = np.array([ (start_time+i) for i in range(0, batch_size)], dtype=np.int32).squeeze()

        inputs = {
            "input_ids": input_ids_batch,
            "input_mask": input_mask_batch,
            "segment_ids": segment_ids_batch,
            "unique_ids": uids
        }
        
        inference_func = self._model.model.signatures["serving_default"]
        outputs = inference_func(**({k: tf.convert_to_tensor(v) for k, v in inputs.items()}))

        return ["results"]
        


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    pipeline_options = PipelineOptions(save_main_session=True)
    question_list = [("""TensorRT is a high performance deep learning inference platform
                                that delivers low latency and high throughput for apps such as
                                recommenders, speech and image/video on NVIDIA GPUs. It includes
                                parsers to import models, and plugins to support novel ops and
                                layers before applying optimizations for inference. Today NVIDIA
                                is open-sourcing parsers and plugins in TensorRT so that the deep
                                learning community can customize and extend these components to
                                take advantage of powerful TensorRT optimizations for your apps.""",
                      ["What is TensorRT?", "Is TensorRT open sourced?", "Who is open sourcing TensorRT?",
                       "What does TensorRT deliver?"] * 4)] * 1000
    engine_path='model.savedmodel'
    with beam.Pipeline(options=pipeline_options) as p:
        shared_handle = shared.Shared()
        _ = (p | beam.Create(question_list)
             | beam.ParDo(DoManualInference(shared_handle=shared_handle, engine_path=engine_path, batch_size=16))
             )
