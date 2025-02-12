import torch
from PIL import Image
import torchvision.transforms.functional as TF
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import os
from PIL import Image, ImageDraw
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from typing import Tuple, List
from torch.utils.data import Dataset



class AspectRatio(Enum):
    RATIO_9_16 = (9, 16)  # 0.5625
    RATIO_3_4 = (3, 4)    # 0.75
    RATIO_1_1 = (1, 1)    # 1.0
    RATIO_4_3 = (4, 3)    # 1.333...
    RATIO_16_9 = (16, 9)  # 1.777...

@dataclass
class ResizeConfig:
    ratio: AspectRatio
    width: int
    height: int

# Define resolution configurations for both 540p and 720p
RESIZE_CONFIGS_540P = {
    AspectRatio.RATIO_9_16: ResizeConfig(AspectRatio.RATIO_9_16, 544, 960),
    AspectRatio.RATIO_3_4: ResizeConfig(AspectRatio.RATIO_3_4, 624, 832),
    AspectRatio.RATIO_1_1: ResizeConfig(AspectRatio.RATIO_1_1, 720, 720),
    AspectRatio.RATIO_4_3: ResizeConfig(AspectRatio.RATIO_4_3, 832, 624),
    AspectRatio.RATIO_16_9: ResizeConfig(AspectRatio.RATIO_16_9, 960, 544),
}

RESIZE_CONFIGS_720P = {
    # 720x1280
    AspectRatio.RATIO_9_16: ResizeConfig(AspectRatio.RATIO_9_16, 704, 1248),
    AspectRatio.RATIO_3_4: ResizeConfig(AspectRatio.RATIO_3_4, 832, 1104),
    AspectRatio.RATIO_1_1: ResizeConfig(AspectRatio.RATIO_1_1, 960, 960),
    AspectRatio.RATIO_4_3: ResizeConfig(AspectRatio.RATIO_4_3, 1104, 832),
    AspectRatio.RATIO_16_9: ResizeConfig(AspectRatio.RATIO_16_9, 1248, 704),
}

class TestImageDataset(Dataset):
    def __init__(
        self,
        image_paths_and_scales: List[tuple],
        high_resolution: bool = False,
        resize: bool = True,
        height: int = 720,
        width: int = 1280,
        crop: bool = False,
    ):
        self.image_paths_and_scales = image_paths_and_scales
        self.resize = resize
        self.resize_configs = RESIZE_CONFIGS_720P if high_resolution else RESIZE_CONFIGS_540P
        print(f"Using {'720p' if high_resolution else '540p'} configuration")
        
    def calculate_aspect_ratio(self, width: int, height: int) -> float:
        """Calculate aspect ratio (width/height)"""
        return width / height

    def __len__(self):
        return len(self.image_paths_and_scales)

    def find_closest_aspect_ratio(self, img_ratio: float) -> Optional[AspectRatio]:
        """Find the closest standard aspect ratio based on absolute difference"""
        standard_ratios = [(ratio, ratio.value[0]/ratio.value[1]) for ratio in AspectRatio]
        standard_ratios.sort(key=lambda x: x[1])
        closest_ratio = min(standard_ratios, key=lambda x: abs(x[1] - img_ratio))
        return closest_ratio[0]

    def process_image(self, image: Image.Image) -> Optional[Tuple[torch.Tensor, AspectRatio]]:
        """
        Process a single image using center cropping to match target aspect ratio
        """
        if not self.resize:
            return TF.to_tensor(image), None

        # Get original dimensions and ratio
        width, height = image.size
        img_ratio = self.calculate_aspect_ratio(width, height)
        target_ratio = self.find_closest_aspect_ratio(img_ratio)
        config = self.resize_configs[target_ratio]
        target_w, target_h = config.width, config.height
        target_aspect = target_w / target_h
        
        # Calculate intermediate size that preserves the larger dimension
        if img_ratio > target_aspect:  # image is wider than target
            new_height = target_h
            new_width = int(new_height * img_ratio)
        else:  # image is taller than target
            new_width = target_w
            new_height = int(new_width / img_ratio)
            
        transform = transforms.Compose([
            transforms.Resize((new_height, new_width)),  # Resize preserving aspect ratio
            transforms.CenterCrop((target_h, target_w)),  # Crop to target size
            transforms.ToTensor(),
        ])
        
        # Apply transformations
        processed_tensor = transform(image)
        
        print(f"Final image ratio: {target_w/target_h:.4f} | width: {target_w} | height: {target_h}")
        return processed_tensor, target_ratio

    def __getitem__(self, idx):
        guidance_audio, guidance_text = -1, -1
        # Parse input parameters
        if len(self.image_paths_and_scales[idx]) == 2:
            img_path, video_length = self.image_paths_and_scales[idx]
            mask_scale, audio_path, prompt = 1.0, "", ""
        elif len(self.image_paths_and_scales[idx]) == 4:
            img_path, video_length, mask_scale, audio_path = self.image_paths_and_scales[idx]
            prompt = ""
        elif len(self.image_paths_and_scales[idx]) == 5:
            img_path, video_length, mask_scale, audio_path, prompt = self.image_paths_and_scales[idx]
        elif len(self.image_paths_and_scales[idx]) == 7:
            img_path, video_length, mask_scale, audio_path, prompt, guidance_audio, guidance_text = self.image_paths_and_scales[idx]
        else:
            raise ValueError("Unsupported format in image_paths_and_scales")


        # Process image
        image = Image.open(img_path).convert("RGB")
        result = self.process_image(image)
        
        if result is None:
            raise ValueError(f"Image {img_path} has unsupported aspect ratio")
            
        processed_image, _ = result

        return {
            "image": processed_image,
            "img_name": os.path.basename(img_path)[:-4],
            "video_length": video_length,
            "mask_scale": mask_scale,
            "audio_path": audio_path,
            "ref_image_path": img_path,
            "prompt": prompt,
            "guidance_audio": guidance_audio,
            "guidance_text": guidance_text,
        }
 
