# python ./scriptss/fid/split_coco.py \
#   --tsv ./nogit/captions/captions_seed-2023_samples-10_duplicates-False.tsv \
#   --inference-path ./nogit/inference/768-v-ema-scale8-ddim-steps50-captions_seed-2023_samples-40000_duplicates-False \
#   --output-path ./nogit/val_splits/10

import os
import argparse
import shutil
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tsv",
    type=str,
)
parser.add_argument(
    "--coco-path",
    type=str,
    default="/datasets/coco2014/val2014"
)
parser.add_argument(
    "--inference-path",
    type=str,
)
parser.add_argument(
    "--output-path",
    type=str,
    default="/tmp"
)

args = parser.parse_args()

# load tsv file
df = pd.read_csv(args.tsv, delimiter="\t")

# Prepare coco_fnames
coco_fnames = df['image_id'].tolist()
coco_fnames = [f"COCO_val2014_{fname:12}.jpg" for fname in coco_fnames]

# Prepare inference_fnames
inference_fnames = df['id'].tolist()
inference_fnames = [f"{fname}.jpg" for fname in inference_fnames]

# if output paths doesn't exist, create it
coco_output_path = os.path.join(args.output_path, "coco")
inference_output_path = os.path.join(args.output_path, "inference")
if not os.path.exists(coco_output_path):
    os.mkdir(coco_output_path)
if not os.path.exists(inference_output_path):
    os.mkdir(inference_output_path)

# copy coco_fnames from argss.coco-path to --output-path
for f in coco_fnames:
    shutil.copy2(os.path.join(args.coco_path, f), os.path.join(coco_output_path, f))

# copy inference_fnames from args.inference-path to --output-path
for f in inference_fnames:
    shutil.copy2(os.path.join(args.inference_path, f), os.path.join(inference_output_path, f))
