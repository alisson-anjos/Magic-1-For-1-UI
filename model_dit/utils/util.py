import importlib
import os
import os.path as osp
import shutil
import sys
from importlib import import_module
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import logging
import os
import random
from typing import Any, Dict, List, Optional

from lightning.fabric.utilities.imports import _NUMPY_AVAILABLE

def instantiate(config: DictConfig, instantiate_module=True):
    """Get arguments from config."""
    module = import_module(config.module_name)
    class_ = getattr(module, config.class_name)
    if instantiate_module:
        init_args = {k: v for k, v in config.items() if k not in ["module_name", "class_name"]}
        return class_(**init_args)
    else:
        return class_


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_videos_from_pil(pil_images, path, fps=8):

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def images_to_video(image_data, output_video_path, fps=25):
    import imageio

    # Convert PIL images to numpy arrays if necessary
    image_data = [np.array(img) if isinstance(img, Image.Image) else img for img in image_data]

    # Use mimwrite to save the video
    imageio.mimwrite(output_video_path, image_data, fps=fps)

    print(f"Video saved to {output_video_path}")


def save_videos_grid(
    videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, text="", image_to_video_with_imageio=False
):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    font = ImageFont.load_default()

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        # Create draw object and font object
        draw = ImageDraw.Draw(x)

        # Write text on the top left corner

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if image_to_video_with_imageio:
        images_to_video(outputs, path, fps)
    else:
        save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def seed_numpy_torch(seed: Optional[int] = None, workers: bool = False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    if _NUMPY_AVAILABLE:
        import numpy as np

        np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
