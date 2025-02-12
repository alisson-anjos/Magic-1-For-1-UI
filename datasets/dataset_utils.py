from pathlib import Path
import joblib
import torch
from einops import repeat, rearrange
import numpy as np

from model_dit.utils.cuda_timer import CudaTimer


def tonp(x):
    return x.detach().cpu().numpy()


### DATA LOADING ###
def parse_vid_id_from_path(vid_path):
    vid_id = (
        str(Path(vid_path).stem)
        .replace('_25fps.mp4','')
        .replace('_origin.mp4','')
        .replace('.mp4','')
        .replace('_25fps','')
        .replace('_origin','')
    )
    return vid_id


def parse_protocol2(data_info):
    data_name_meta = []
    mp4_suffix = data_info.mp4_suffix
    wav_suffix = data_info.wav_suffix
    meta_pt = data_info.meta_pt
    root = data_info.root
    data_meta = torch.load(meta_pt)
    for meta in data_meta:
        vid_id = meta['vid_id']
        vid_id = parse_vid_id_from_path(vid_id)
        vid_id = vid_id.split('/')[-1]
        # filter error id
        if vid_id in []:
            continue
        data_name_meta.append({
            'video_path': str(Path(root,f'{vid_id}{mp4_suffix}.mp4')),
            'wav_path': str(Path(root,f'{vid_id}{wav_suffix}.wav')),
            'bbox': meta['bbox'],
        })   
    return data_name_meta  


### FPS UTILS ###
def convert_var_to_nay_fps(var, ori_var_fps, target_fps):
    """
    Args:
        var: (T,...)
        ori_var_fps: 25
    """
    original_frame_count = len(var)
    target_frame_count = int((original_frame_count / ori_var_fps) * target_fps)
    target_frame_indices = [
        min(
            round((target_fps_pos / target_fps) * ori_var_fps),
            original_frame_count - 1,
        )
        for target_fps_pos in range(target_frame_count)
    ]
    target_frame_indices = np.array(target_frame_indices)
    return var[target_frame_indices]


### SAMPLING UTILS ###
def get_dataset_sample_rate(dset_sample_rate, target_rate):
        if target_rate is None:
            return dset_sample_rate
        target_rate = np.array(target_rate)
        
        dset_sample_rate_new = (dset_sample_rate * target_rate)
        dset_sample_rate_new = dset_sample_rate_new / np.sum(dset_sample_rate_new)
        return dset_sample_rate_new
