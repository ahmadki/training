import argparse, os
import cv2
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
try:
    from lightning.pytorch import seed_everything
except:
    from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
# from scripts.utils import replace_module, getModelSize

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="rank of the process",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of processes",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="index of GPU to use",
    )
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
        default="id",
        help="column name of the output image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--use-int8",
        type=bool,
        default=False,
        help="use int8 for inference",
    )
    parser.add_argument(
        "--use-ema",
        type=bool,
        default=None,
        help="override use_ema in model config",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    # TODO(ahmadki): force ema=False ?
    if opt.use_ema is not None:
        config.model.params.use_ema = opt.use_ema
    model = load_model_from_config(config, f"{opt.ckpt}")

    # TODO(ahmadki): clean up this part
    if torch.cuda.is_available():
        device = torch.device("cuda") # if opt.gpu is None else torch.cuda.device(opt.gpu)
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.output_dir, exist_ok=True)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    print(f"reading prompts from {opt.inference_tsv}")
    df = pd.read_csv(opt.inference_tsv, delimiter="\t")
    fnames = df[opt.fname_col].tolist()
    prompts = df[opt.prompt_col].tolist()

    print(f"splitting prompts among {opt.world_size} ranks")
    fnames = fnames[opt.rank::opt.world_size]
    prompts = prompts[opt.rank::opt.world_size]

    print(f"batching the dataset")
    batched_fnames = list(chunk(fnames, opt.batch_size))
    batched_prompts = list(chunk(prompts, opt.batch_size))
    batched_data = list(zip(batched_fnames, batched_prompts))

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    sample_count = 0
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():

            for batch in tqdm(batched_data, desc="data"):
                batch_fnames = batch[0]
                batch_prompts = batch[1]
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(len(batch_prompts) * [""])
                c = model.get_learned_conditioning(batch_prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples, _ = sampler.sample(S=opt.steps,
                                            conditioning=c,
                                            batch_size=len(batch_prompts),
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for fname, x_sample in zip(batch_fnames, x_samples):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(opt.output_dir, f"{fname}.png"))
                    sample_count += 1

    print(f"rank {opt.rank} of {opt.world_size} processed {sample_count} samples.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
