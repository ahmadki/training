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

# NOTE: requires pytorch to write output file, so run something like:
# docker run --env PYTHONDONTWRITEBYTECODE=1 -v $(pwd):/scratch \
#        --ipc=host nvcr.io/nvidia/pytorch:20.08-py3            \
#        /scratch/numpy-to-torch.py /scratch/resnet34-333f7ec4.pickle  /scratch/resnet34-333f7ec4.pth

import torch
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="read in pickled numpy file and convert to pytorch .pth file")

parser.add_argument('input_file', type=str, help='input pickled numpy file')
parser.add_argument('output_file', type=str, help='output pytorch .pth file')
parser.add_argument('--verbose', action='store_true',
                    help='also print the list of layers and their sizes')

conversion_dict = {
    # use None instead of string for output_name to keep the layer from being outputted
#    'input_name': 'output_name'
}

def convert_numpy_name_to_pytorch_name(name, shape):
    # pytorch calls batch-norm beta and gamma, "bias" and "weight" respectively:
    if len(shape) == 1:
        name = name.replace("beta", "bias")
        name = name.replace("gamma", "weight")

    # use the input name as output name by default, but if we have an explicit
    # mapping in the conversion_dict, then use it
    return conversion_dict.get(name, name)
    
args = parser.parse_args()

with open(args.input_file, 'rb') as numpy_file:
    numpy_dict = pickle.load(numpy_file)

output_dict = {}

for input_name, value in numpy_dict.items():
    output_name = convert_numpy_name_to_pytorch_name(input_name, value.shape)
    if output_name is None:
        continue
    
    if args.verbose:
        print(input_name, output_name, value.shape)

    output_dict[output_name] = torch.from_numpy(value)

with open(args.output_file, 'wb') as out_file:
    torch.save(output_dict, out_file)
