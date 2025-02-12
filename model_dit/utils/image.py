import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import pad


def tensor2image(
    tensor: torch.Tensor,  # C, H, W
):
    tensor = (tensor * 255).byte()
    if tensor.ndim == 4:
        print(f"{tensor.shape=}")
        tensor = tensor.squeeze(1)
    array = tensor.permute(1, 2, 0).numpy()
    return array

def crop(img_array, bbox):
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    img_crop = img_array[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :]
    return img_crop

def tonp(x):
    return x.detach().cpu().numpy()

def crop_and_resize(images, new_size, original_size, start):
    cropped_image = transforms.functional.crop(images, start[1], start[0], new_size, new_size)
    resized_image = transforms.Resize((original_size, original_size))(cropped_image)
    return resized_image

def resize_and_pad(images, new_size, padding, pixel_trans=1):
    resized_image = transforms.Resize((new_size, new_size))(images)
    if pixel_trans:
        padded_image = pad(resized_image, padding, fill=0.0, padding_mode="edge")
    else:
        padded_image = pad(resized_image, padding, fill=0.0, padding_mode="constant")
    return padded_image

def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    sw *= scale
    sh *= scale
    scale_bbox = [cx - sw, cy - sh, cx + sw, cy + sh]
    scale_bbox[0] = np.clip(scale_bbox[0], 0, w)
    scale_bbox[2] = np.clip(scale_bbox[2], 0, w)
    scale_bbox[1] = np.clip(scale_bbox[1], 0, h)
    scale_bbox[3] = np.clip(scale_bbox[3], 0, h)
    return scale_bbox

def mediapipe2s3fd(bbox, extension_factor=0.1):
    bbox_height = bbox[3] - bbox[1]
    bbox_scale_up_value = extension_factor * bbox_height
    bbox[1] = int(max(0, bbox[1] - bbox_scale_up_value))
    return bbox

def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise Exception("Invalid mask")
    # sontime bbox is like this: array([ -8.84635544, 216.97692871, 192.20074463, 502.83700562])
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    # tgt_pose = np.zeros_like(tgt_img.asnumpy())
    tgt_pose = np.zeros((hd, wd, 3))
    tgt_pose[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :] = 255.0
    if return_pil:
        tgt_pose_pil = Image.fromarray(tgt_pose.astype(np.uint8))
        return tgt_pose_pil
    return tgt_pose

def get_move_area(bbox, fw, fh):
    move_area_bbox = [
        bbox[:, 0].min(),
        bbox[:, 1].min(),
        bbox[:, 2].max(),
        bbox[:, 3].max(),
    ]
    if move_area_bbox[0] < 0:
        move_area_bbox[0] = 0
    if move_area_bbox[1] < 0:
        move_area_bbox[1] = 0
    if move_area_bbox[2] > fw:
        move_area_bbox[2] = fw
    if move_area_bbox[3] > fh:
        move_area_bbox[3] = fh
    return move_area_bbox

def crop_np(img_array, bbox):
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(ii) for ii in bbox]
    img_crop = img_array[bbox_y0:bbox_y1, bbox_x0:bbox_x1, :]
    return img_crop

def xyxy_to_xys(det):
    s = max((det[3] - det[1]), (det[2] - det[0])) / 2
    y = (det[1] + det[3]) / 2  # crop center x
    x = (det[0] + det[2]) / 2  # crop center y
    return [x, y, s]

def xys_to_xyxy(x, y, s, scale=3.0):
    s *= scale
    return np.stack([x - s, y - s, x + s, y + s], axis=-1)

def get_rand_s(width, height, bbox):
    bbox_s = bbox[2]
    max_s = min(min(bbox[0], width - bbox[0]) / (bbox_s), min(bbox[1], height - bbox[1]) / (bbox_s))
    if max_s < 1:
        max_s = 1
    if max_s > 2.5:
        max_s = 2.5
    s = max_s
    return s

def get_scale_bbox(wd, hd, union_bbox):
    union_bbox_xys = xyxy_to_xys(union_bbox)
    scale = get_rand_s(wd, hd, union_bbox_xys)
    scale_bbox = xys_to_xyxy(*union_bbox_xys, scale=scale)
    return scale_bbox