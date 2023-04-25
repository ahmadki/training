#!/usr/bin/env python3

import os
import json
import argparse

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input-captions-file", type=str, required=True)
parser.add_argument("--output-tsv", type=str, default=None)
parser.add_argument("--num-samples", type=int, default=30000)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--allow-duplicate-images", type=bool, default=False)

args = parser.parse_args()

# Load coco annotations
with open(args.input_captions_file, "r") as f:
    captions = json.load(f)
    annotations = captions["annotations"]

# Convert to dataframe
df = pd.DataFrame(annotations)
df['caption'] = df['caption'].apply(lambda x: x.replace('\n', '').strip())

# Shuffle the dataframe
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Keep a single captions per image
if not args.allow_duplicate_images:
    df = df.drop_duplicates(subset=["image_id"], keep="first")

# Take a subset
df = df[:args.num_samples]

# Sort by id
df = df.sort_values(by=["id"])

# Save the subset to a tsv file
output_fname = args.output_tsv if args.output_tsv else f"captions_seed-{args.seed}_samples-{args.num_samples}_duplicates-{args.allow_duplicate_images}.tsv"
df.to_csv(output_fname, sep="\t", index=False)
