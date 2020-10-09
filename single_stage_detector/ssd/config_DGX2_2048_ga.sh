#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --weight-decay       "4e-5"
               --lr                 "0.005"
               --warmup             "30"
               --lr-decay-schedule  "60 75"
               --batch-size         "128"
               --batch-splits       "4"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=1:00:00

## System config params
DGXNGPU=16
DGXSOCKETCORES=20
DGXNSOCKET=2
DGXHT=2     # HT is on is 2, HT off is 1
DGXIBDEVICES='--device=/dev/infiniband --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/umad3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/umad2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/umad1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0'
