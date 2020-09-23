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

# NOTE: requires pytorch to read input file, so run something like:
# docker run --env PYTHONDONTWRITEBYTECODE=1 -v $(pwd):/scratch \
#        --ipc=host nvcr.io/nvidia/pytorch:20.08-py3            \
#        /scratch/torch-to-numpy.py /scratch/resnet34-333f7ec4.pth  /scratch/resnet34-333f7ec4.pickle

import torch
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="read in pytorch .pth file and convert to pickled dictionary of numpy arrays")

parser.add_argument('input_file', type=str, help='input pytorch .pth file name')
parser.add_argument('output_file', type=str, help='name of output pickled python file with dictionary of numpy arrays')
parser.add_argument('--verbose', action='store_true',
                    help='also print the list of tensors and their sizes')

args = parser.parse_args()

with open(args.input_file, 'rb') as pth_file:
    pt = torch.load(pth_file)

output_dict = {}

for name, value in pt.items():
    if "fc" in name:
        # if we're just using a backbone then the fully connected layer from
        # the end of the original model is irrelevant (and huge), so leave it
        # out of the pickle file
        if args.verbose:
            print("ignoring fully connected layer", name)
        continue
        
    numpy_array = value.data.numpy()
    assert numpy_array.dtype == 'float32'

    # pytorch uses unusual names for batchnorm beta and gamma, so convert to
    # something more recognizable:
    if len(numpy_array.shape) == 1:
        if "bias" in name:
            name = name.replace("bias", "beta")
        if "weight" in name:
            name = name.replace("weight", "gamma")

    output_dict[name] = numpy_array

    if args.verbose:
        print(name, output_dict[name].shape)

with open(args.output_file, 'wb') as out_file:
    pickle.dump(output_dict, out_file)
