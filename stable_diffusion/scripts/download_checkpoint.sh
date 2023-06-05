#!/bin/bash

: "${DOWNLOAD_PATH:=/checkpoints/stability}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
    shift
done

mkdir -p ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

wget -c https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt

echo "	d635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824  ./512-base-ema.ckpt" | sha256sum -c
