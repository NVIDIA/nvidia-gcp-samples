#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Usage:
    
    Context and questions are to be put into json file respecting SQuAD format.
    ex)

    python get_request_body_bert.py -f <filename>.json

    curl \
    -X GET -k -H "Content-Type: application/json" \
    -H "Authorization: Bearer `gcloud auth print-access-token`" \
    "${ENDPOINT}/projects/${PROJECT_NAME}/models/${MODEL_NAME}"
    
    curl \
    -X POST <triton_host_ip>:8000/v2/models/bert/infer \
    -k -H "Content-Type: application/json" \
    -d @bert_squad.json

    expected output: 

"""
from utils.create_squad_data import read_squad_examples, convert_examples_to_features

import argparse
import json
import numpy as np
import os
import random
import run_squad
import sys 
import struct
import tensorflow as tf 
import time 
import tokenization 

class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.
    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.
    """

    def __init__(self, name, shape, datatype):
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._parameters = {}
        self._data = None
        self._raw_data = None

    def name(self):
        """Get the name of input associated with this object.
        Returns
        -------
        str
            The name of input
        """
        return self._name

    def datatype(self):
        """Get the datatype of input associated with this object.
        Returns
        -------
        str
            The datatype of input
        """
        return self._datatype

    def shape(self):
        """Get the shape of input associated with this object.
        Returns
        -------
        list
            The shape of input
        """
        return self._shape

    def set_shape(self, shape):
        """Set the shape of input.
        Parameters
        ----------
        shape : list
            The shape of the associated input.
        """
        self._shape = shape

    def set_data_from_numpy(self, input_tensor, binary_data=True):
        """Set the tensor data from the specified numpy array for
        input associated with this object.
        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        binary_data : bool
            Indicates whether to set data for the input in binary format
            or explicit tensor within JSON. The default value is True,
            which means the data will be delivered as binary data in the
            HTTP body after the JSON object.
        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        dtype = np_to_triton_dtype(input_tensor.dtype)
        if self._datatype != dtype:
            raise_error(
                "got unexpected datatype {} from numpy array, expected {}".
                format(dtype, self._datatype))
        valid_shape = True
        if len(self._shape) != len(input_tensor.shape):
            valid_shape = False
        else:
            for i in range(len(self._shape)):
                if self._shape[i] != input_tensor.shape[i]:
                    valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1],
                    str(self._shape)[1:-1]))

        self._parameters.pop('shared_memory_region', None)
        self._parameters.pop('shared_memory_byte_size', None)
        self._parameters.pop('shared_memory_offset', None)

        if not binary_data:
            self._parameters.pop('binary_data_size', None)
            self._raw_data = None
            if self._datatype == "BYTES":
                self._data = [val for val in input_tensor.flatten()]
            else:
                self._data = [val.item() for val in input_tensor.flatten()]
        else:
            self._data = None
            if self._datatype == "BYTES":
                self._raw_data = serialize_byte_tensor(input_tensor).tobytes()
            else:
                self._raw_data = input_tensor.tobytes()
            self._parameters['binary_data_size'] = len(self._raw_data)

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.
        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.
        """
        self._data = None
        self._raw_data = None
        self._parameters.pop('binary_data_size', None)

        self._parameters['shared_memory_region'] = region_name
        self._parameters['shared_memory_byte_size'] = byte_size
        if offset != 0:
            self._parameters['shared_memory_offset'].int64_param = offset

    def _get_binary_data(self):
        """Returns the raw binary data if available
        Returns
        -------
        bytes
            The raw data for the input tensor
        """
        return self._raw_data

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.
        Returns
        -------
        dict
            The underlying tensor specification as dict
        """
        if self._parameters.get('shared_memory_region') is not None or \
                self._raw_data is not None:
            return {
                'name': self._name,
                'shape': self._shape,
                'datatype': self._datatype,
                'parameters': self._parameters,
            }
        else:
            return {
                'name': self._name,
                'shape': self._shape,
                'datatype': self._datatype,
                'parameters': self._parameters,
                'data': self._data
            }

class InferRequestedOutput:
    """An object of InferRequestedOutput class is used to describe a
    requested output tensor for an inference request.
    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object.
    binary_data : bool
        Indicates whether to return result data for the output in
        binary format or explicit tensor within JSON. The default
        value is True, which means the data will be delivered as
        binary data in the HTTP body after JSON object. This field
        will be unset if shared memory is set for the output.
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name, binary_data=True, class_count=0):
        self._name = name
        self._parameters = {}
        if class_count != 0:
            self._parameters['classification'] = class_count
        self._binary = binary_data
        self._parameters['binary_data'] = binary_data

    def name(self):
        """Get the name of output associated with this object.
        Returns
        -------
        str
            The name of output
        """
        return self._name

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Marks the output to return the inference result in
        specified shared memory region.
        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.
        """
        if 'classification' in self._parameters:
            raise_error("shared memory can't be set on classification output")
        if self._binary:
            self._parameters['binary_data'] = False

        self._parameters['shared_memory_region'] = region_name
        self._parameters['shared_memory_byte_size'] = byte_size
        if offset != 0:
            self._parameters['shared_memory_offset'] = offset

    def unset_shared_memory(self):
        """Clears the shared memory option set by the last call to
        InferRequestedOutput.set_shared_memory(). After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

        self._parameters['binary_data'] = self._binary
        self._parameters.pop('shared_memory_region', None)
        self._parameters.pop('shared_memory_byte_size', None)
        self._parameters.pop('shared_memory_offset', None)

    def _get_tensor(self):
        """Retrieve the underlying input as json dict.
        Returns
        -------
        dict
            The underlying tensor as a dict
        """
        return {'name': self._name, 'parameters': self._parameters}

class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.
    Parameters
    ----------
    result : dict
        The inference response from the server
    verbose : bool
        If True generate verbose output. Default value is False.
    """

    def __init__(self, response, verbose):
        header_length = response.get('Inference-Header-Content-Length')
        header_length = None
        if header_length is None:
            content = response.read()
            if verbose:
                print(content)
            self._result = json.loads(content)
        else:
            header_length = int(header_length)
            content = response.read(length=header_length)
            if verbose:
                print(content)
            self._result = json.loads(content)

            # Maps the output name to the index in buffer for quick retrieval
            self._output_name_to_buffer_map = {}
            # Read the remaining data off the response body.
            self._buffer = response.read()
            buffer_index = 0
            for output in self._result['outputs']:
                parameters = output.get("parameters")
                if parameters is not None:
                    this_data_size = parameters.get("binary_data_size")
                    if this_data_size is not None:
                        self._output_name_to_buffer_map[
                            output['name']] = buffer_index
                        buffer_index = buffer_index + this_data_size

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format
        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.
        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        if self._result.get('outputs') is not None:
            for output in self._result['outputs']:
                if output['name'] == name:
                    datatype = output['datatype']
                    has_binary_data = False
                    parameters = output.get("parameters")
                    if parameters is not None:
                        this_data_size = parameters.get("binary_data_size")
                        if this_data_size is not None:
                            has_binary_data = True
                            if this_data_size != 0:
                                start_index = self._output_name_to_buffer_map[
                                    name]
                                end_index = start_index + this_data_size
                                if datatype == 'BYTES':
                                    # String results contain a 4-byte string length
                                    # followed by the actual string characters. Hence,
                                    # need to decode the raw bytes to convert into
                                    # array elements.
                                    np_array = deserialize_bytes_tensor(
                                        self._buffer[start_index:end_index])
                                else:
                                    np_array = np.frombuffer(
                                        self._buffer[start_index:end_index],
                                        dtype=triton_to_np_dtype(datatype))
                            else:
                                np_array = np.empty(0)
                    if not has_binary_data:
                        np_array = np.array(output['data'],
                                            dtype=triton_to_np_dtype(datatype))
                    np_array = np.resize(np_array, output['shape'])
                    return np_array
        return None

    def get_output(self, name):
        """Retrieves the output tensor corresponding to the named ouput.
        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.
        Returns
        -------
        Dict
            If an output tensor with specified name is present in
            the infer resonse then returns it as a json dict,
            otherwise returns None.
        """
        for output in self._result['outputs']:
            if output['name'] == name:
                return output

        return None

    def get_response(self):
        """Retrieves the complete response
        Returns
        -------
        dict
            The underlying response dict.
        """
        return self._result

def np_to_triton_dtype(np_dtype):
    if np_dtype == np.bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object or np_dtype.type == np.bytes_:
        return "BYTES"
    return None

def triton_to_np_dtype(dtype):
    if dtype == "BOOL":
        return np.bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.object
    return None

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        label_ids_data = ()
        input_ids_data = ()
        input_mask_data = ()
        segment_ids_data = ()
        for i in range(0, min(n, l-ndx)):
            label_ids_data = label_ids_data + (np.array([iterable[ndx + i].unique_id], dtype=np.int32),)
            input_ids_data = input_ids_data+ (np.array(iterable[ndx + i].input_ids, dtype=np.int32),)
            input_mask_data = input_mask_data+ (np.array(iterable[ndx + i].input_mask, dtype=np.int32),)
            segment_ids_data = segment_ids_data+ (np.array(iterable[ndx + i].segment_ids, dtype=np.int32),)

        inputs_dict = {'unique_ids': label_ids_data,
                       'input_ids': input_ids_data,
                       'input_mask': input_mask_data,
                       'segment_ids': segment_ids_data}
        return inputs_dict

def get_bert_inference_request(inputs, request_id, outputs, sequence_id,
                           sequence_start, sequence_end, priority, timeout):
    infer_request = {}
    parameters = {}
    if request_id != "":
        infer_request['id'] = request_id
    if sequence_id != 0:
        parameters['sequence_id'] = sequence_id
        parameters['sequence_start'] = sequence_start
        parameters['sequence_end'] = sequence_end
    if priority != 0:
        parameters['priority'] = priority
    if timeout is not None:
        parameters['timeout'] = timeout

    infer_request['inputs'] = [
        this_input._get_tensor() for this_input in inputs
    ]
    if outputs:
        infer_request['outputs'] = [
            this_output._get_tensor() for this_output in outputs
        ]

    if parameters:
        infer_request['parameters'] = parameters

    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)
        return request_body, json_size
    return request_body, None

def get_bert_request_body(input_data, version_2_with_negative, 
                          tokenizer, max_seq_length, doc_stride,
                          max_query_length):
    eval_examples = read_squad_examples(input_file=None, is_training=False, 
                                    version_2_with_negative=version_2_with_negative, 
                                    input_data=input_data)
    eval_features = []
    def append_feature(feature):
        eval_features.append(feature) 
    
    convert_examples_to_features(
        examples=eval_examples[0:],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature)

    inputs_dict = batch(eval_features)
    label_ids_data = np.stack(inputs_dict['unique_ids'])
    input_ids_data = np.stack(inputs_dict['input_ids'])
    input_mask_data = np.stack(inputs_dict['input_mask'])
    segment_ids_data = np.stack(inputs_dict['segment_ids'])

    inputs = []
    inputs.append(InferInput('input_ids', input_ids_data.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_ids_data, binary_data=False)
    inputs.append(InferInput('input_mask', input_mask_data.shape, "INT32"))
    inputs[1].set_data_from_numpy(input_mask_data, binary_data=False)
    inputs.append(InferInput('segment_ids', segment_ids_data.shape, "INT32"))
    inputs[2].set_data_from_numpy(segment_ids_data, binary_data=False)

    outputs = []
    outputs.append(InferRequestedOutput('cls_squad_logits', binary_data=False))

    request_id = ''
    sequence_id = 0
    sequence_start = False
    sequence_end = False
    priority = 0
    timeout = None
    headers=None
    query_params=None

    request_body, json_size = get_bert_inference_request(
                                inputs=inputs,
                                request_id=request_id,
                                outputs=outputs,
                                sequence_id=sequence_id,
                                sequence_start=sequence_start,
                                sequence_end=sequence_end,
                                priority=priority,
                                timeout=timeout)
    return request_body, inputs_dict, eval_examples, eval_features