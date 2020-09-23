#!/usr/bin/env python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


# NOTE: requires TensorFlow to read input file, so run something like:
# docker run --env PYTHONDONTWRITEBYTECODE=1 -v $(pwd):/scratch \
#        --ipc=host nvcr.io/nvidia/tensorflow:20.08-tf1-py3            \
#        /scratch/tf-to-numpy.py /scratch/model.ckpt-12345  /scratch/resnet34-333f7ec4.pickle


# info about how to read tf chkpoints and convert them is based on
# https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28

import os
import pickle
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np

parser = ArgumentParser(description='read in TensorFlow checkpoint and convert to pickled dictionary of numpy arrays')

parser.add_argument('input_file', type=str,
                    help='input TensorFlow checkpoint name, usually something like "model.ckpt-12345"')
parser.add_argument('output_file', type=str,
                    help='name of output pickled python file with dictionary of numpy arrays')
parser.add_argument('--verbose', action='store_true',
                    help='also print the list of tensors and their sizes')

args = parser.parse_args()

tf_path=os.path.abspath(args.input_file)
tf_vars = tf.train.list_variables(tf_path)

output_dict = {}

for name, shape in tf_vars:
    if "Momentum" in name or "global_step" in name:
        # ignore optimizer state, we just want model weights
        continue

    if "dense" in name:
        # if we're just using a backbone then the dense-layer from the end of
        # the original model is irrelevant (and huge), so leave it out of the
        # pickle file
        if args.verbose:
            print("ignoring dense layer", name)
        continue
            
    numpy_array = tf.train.load_variable(tf_path, name)
    assert numpy_array.dtype == 'float32'
    
    if len(numpy_array.shape) == 4:
        # tensorflow uses R,S,C,K, while everyone else uses K,C,R,S
        numpy_array = np.transpose(numpy_array, (3, 2, 0, 1))

    # TensorFlow uses unusual names for batchnorm running_mean and var, so
    # convert to something more recognizable:
    if len(numpy_array.shape) == 1:
        if "moving" in name:
            name = name.replace("moving", "running")
        if "variance" in name:
            name = name.replace("variance", "var")

    output_dict[name] = numpy_array
    if args.verbose:
        print(name, output_dict[name].shape)

with open(args.output_file, 'wb') as out_file:
    pickle.dump(output_dict, out_file)
