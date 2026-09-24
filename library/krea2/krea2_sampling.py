"""Functional flow-matching sampler for the K2 MMDiT.
"""

import gc
import math

import torch
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm


def roundup(value, multiple, name):
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


def gather_valid_text(txt, mask):
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]
    max_len = max(v.shape[0] for v in valid)
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((imgmask, txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@torch.no_grad()
def encode_prompts(encoder, prompts, negative_prompts=None, *, cfg=True):
    txt, txtmask = encoder(prompts)
    txt, txtmask = gather_valid_text(txt, txtmask)

    untxt = untxtmask = None
    if cfg:
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        untxt, untxtmask = encoder(negative_prompts)
        untxt, untxtmask = gather_valid_text(untxt, untxtmask)

    return txt, txtmask, untxt, untxtmask


@torch.no_grad()
def sample(
    model,
    ae,
    txt,
    txtmask,
    *,
    untxt=None,
    untxtmask=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    cfg_scale=5.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
):
    patch = model.config.patch

    compression = 2 ** len(ae.temperal_downsample)
    channels = ae.z_dim

    align = compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = txt.shape[0]
    cfg = cfg_scale > 1.0 and untxt is not None

    txt, txtmask = txt.to(device=device, dtype=dtype), txtmask.to(device)
    if cfg:
        untxt, untxtmask = untxt.to(device=device, dtype=dtype), untxtmask.to(device)

    noise = torch.cat(
        [
            torch.randn(
                1,
                channels,
                height // compression,
                width // compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (compression * patch)) ** 2
    x2 = (maxres // (compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    img = x
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=dtype):
        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="sampling"):
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                # Krea's public sampler defines guidance as an increment over
                # the conditional prediction (guidance=0 means plain cond),
                # rather than the conventional uncond-based CFG scale.
                v = cond + cfg_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (compression * patch),
        w=width // (compression * patch),
    )
    original_ae_device = next(ae.parameters()).device
    try:
        ae = ae.to(img.device)
        pixels = ae.decode_to_pixels(img.to(torch.bfloat16))
        pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
        return [Image.fromarray(pixels[i]) for i in range(len(pixels))]
    finally:
        # Always release the decoder and its large temporary activations. This
        # also runs when conversion or image saving raises an exception.
        ae.to(original_ae_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
