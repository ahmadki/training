import os
import argparse

import torch
import torch.nn as nn

import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm


class CLIPEncoder(nn.Module):
    def __init__(self, clip_version='ViT-B/32', pretrained='', cache_dir=None, device='cuda'):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == 'ViT-H-14':
                self.pretrained = 'laion2b_s32b_b79k'
            elif self.clip_version == 'ViT-g-14':
                self.pretrained = 'laion2b_s12b_b42k'
            else:
                self.pretrained = 'openai'

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.clip_version,
                                                                               pretrained=self.pretrained,
                                                                               cache_dir=cache_dir)

        self.model.eval()
        self.model.to(device)

        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if isinstance(image, str):  # filenmae
            image = Image.open(image)
        if isinstance(image, Image.Image):  # PIL Image
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = open_clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-tsv",
        required=True,
        type=str,
        help="load prompts and output file name from this tsv",
    )
    parser.add_argument(
        "--prompt-col",
        type=str,
        default="caption",
        help="column name of prompt",
    )
    parser.add_argument(
        "--fname-col",
        type=str,
        default="image_id",
        help="column name of the output image",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./outputs",
        help="dir to write results to"
    )
    parser.add_argument(
        "--clip-version",
        type=str,
        default="ViT-H-14",
        help="CLIP version to use"
    )

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()

    print('Init CLIP Encoder..')
    encoder = CLIPEncoder(clip_version=opt.clip_version)


    print('Reading tsv..')
    df = pd.read_csv(opt.inference_tsv, delimiter="\t")
    fnames = df[opt.fname_col].tolist()
    prompts = df[opt.prompt_col].tolist()

    print(f'Number of image-text pairs: {len(prompts)}')

    clip_score = 0.
    count = 0
    for fname, prompt in zip(tqdm(fnames), prompts):
        # img = os.path.join(opt.images_dir, f"COCO_val2014_{fname:012}.jpg")
        img = os.path.join(opt.images_dir, f"{fname}.png")
        clip_score += encoder.get_clip_score(prompt, img)
        count += 1

    clip_score = clip_score.flatten().cpu().numpy()[0]
    avg_clip_score = clip_score / count
    print(f'CLIP score: {avg_clip_score}')
