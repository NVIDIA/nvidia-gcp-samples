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

from typing import List, Any, Text, Tuple
import time
import logging

import apache_beam as beam
from apache_beam.utils import shared
from apache_beam.options.pipeline_options import PipelineOptions


class TrtModel():
    def __init__(self, infer_context, vocab_file="vocab.txt"):
        import helpers.tokenization as tokenization

        self.infer_context = infer_context
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
        from polygraphy.backend.trt import EngineFromBytes
        from polygraphy.backend.trt import TrtRunner
        # setup is a good place to initialize transient in-memory resources.
        def initialize_model():
            # Load a potentially large model in memory. Executed once per process.
            build_engine = EngineFromBytes(open(self._engine_path, "rb").read())
            runner = TrtRunner(build_engine)
            runner.activate()
            return TrtModel(runner)

        self._trtModel = self._shared_handle.acquire(initialize_model)

    def process(self, element: Tuple[Text, List[Text]]) -> List[Any]:
        yield (self.predict(element))

    def predict(self, inputs: Tuple[Text, List[Text]]) -> List[Any]:
        import helpers.data_processing as dp
        from polygraphy.backend.trt import TrtRunner
        import numpy as np
        import collections
        import time

        def question_features(tokens, question):
            # Extract features from the paragraph and question
            return dp.convert_example_to_features(tokens, question,
                                                  self._trtModel.tokenizer,
                                                  self._trtModel.max_seq_length,
                                                  self._trtModel.doc_stride,
                                                  self._trtModel.max_query_length)

        features = []
        doc_tokens = dp.convert_doc_tokens(inputs[0])
        ques_list = inputs[1]

        batch_size = len(ques_list)
        if batch_size < 16:
            # Pad the input batch to batch_size to match the model expected input.
            pad = [ques_list[0]] * (16 - batch_size)
            ques_list.extend(pad)

        for question_text in ques_list:
            features.append(question_features(doc_tokens, question_text)[0])

        input_ids_batch = np.dstack([feature.input_ids for feature in features]).squeeze()
        segment_ids_batch = np.dstack([feature.segment_ids for feature in features]).squeeze()
        input_mask_batch = np.dstack([feature.input_mask for feature in features]).squeeze()

        inputs = {
            "input_ids": input_ids_batch,
            "input_mask": input_mask_batch,
            "segment_ids": segment_ids_batch
        }
        output = self._trtModel.infer_context.infer(inputs)

        start_logits = output['cls_squad_logits'][:, :, 0, :, :]
        end_logits = output['cls_squad_logits'][:, :, 1, :, :]
        networkOutputs = [self._NetworkOutput(
            start_logits=start_logits[i, :],
            end_logits=end_logits[i, :],
            feature_index=0) for i in range(self._batch_size)]
        predictions = []
        for feature, networkOutput in zip(features, networkOutputs):
            prediction, _, _ = dp.get_predictions(doc_tokens, [feature],
                                                  [networkOutput], self._trtModel.n_best_size,
                                                  self._trtModel.max_answer_length)
            predictions.append(prediction)

        return ["[Q]: " + ques + "     [A]:" + prediction for ques, prediction in zip(ques_list, predictions)]


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
                       "What does TensorRT deliver?"] * 4)] * 40000
    engine_path = "/workspace/trt_beam/bert_large_seq384_bs16_trt2011.engine"

    start_time = time.time()
    with beam.Pipeline(options=pipeline_options) as p:
        shared_handle = shared.Shared()
        _ = (p | beam.Create(question_list)
             | beam.ParDo(DoManualInference(shared_handle=shared_handle, engine_path=engine_path, batch_size=16))
             | beam.Map(print)
             )
    logging.info(f"--- {time.time() - start_time} seconds ---")
    logging.info(f"--- {len(question_list) * 16.0 // (time.time() - start_time)} questions/seconds ---")
