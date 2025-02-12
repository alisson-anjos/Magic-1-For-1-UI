from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

from diffusers.models import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention import Attention 
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.configuration_utils import ConfigMixin, register_to_config

from lightning.pytorch.utilities import rank_zero_info

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, get_temporal_window_pos_embed, get_3d_sincos_pos_embed, ScaledSinusoidalEmbedding


USE_FLASH_ATTENTION3 = os.getenv('USE_FLASH_ATTENTION3', '0').lower() in ('1', 'true', 'yes')
if USE_FLASH_ATTENTION3:
    from .attenion_flashatt3 import attention, get_cu_seqlens, get_ca_cu_seqlens_audio, get_ca_cu_seqlens_audio_different_tokens
else:
    from .attenion import attention, get_cu_seqlens, get_ca_cu_seqlens 
# DEBUG ring attention

from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from .posemb_layers import get_nd_rotary_pos_embed, get_1d_rotary_pos_embed


def sst_load_state_dict(pretrained_model_path):

    def modify_state_dict(state_dict):  # TODO (DMD2): debug
        new_state_dict = {}
        crucial_key = ["vae", "module"]     
        crucial_dict = [dict(), dict()]
        if any("hydit_unimodel.guidance_model.feedforward_model." in key for key in state_dict.keys()):
            tag = 1
        else:
            tag = 0
        for key, value in state_dict.items():
            if tag == 1:
                if "hydit_unimodel.guidance_model.feedforward_model." in key:
                    new_key = key.replace("hydit_unimodel.guidance_model.feedforward_model.", "")
                    crucial_dict[1][new_key] = value
                elif "vae." in key:
                    new_key = key.replace("vae.", "")
                    crucial_dict[0][new_key] = value  
                else:
                    new_state_dict[key] = value
            else:
                if "model." in key:
                    new_key = key[6:]
                    crucial_dict[1][new_key] = value
                elif "vae." in key:
                    new_key = key.replace("vae.", "")
                    crucial_dict[0][new_key] = value  
                else:
                    new_state_dict[key] = value
        new_state_dict[crucial_key[0]] = crucial_dict[0]
        new_state_dict[crucial_key[1]] = crucial_dict[1]
        return new_state_dict

    load_keys = ["state_dict", "module"]

    # TODO rewrite this lmao 

    if os.path.isfile(pretrained_model_path):
        model_path = pretrained_model_path
        state_dict = torch.load(model_path, map_location="cpu")
    elif os.path.isfile(os.path.join(pretrained_model_path, "mp_rank_00_model_states.pt")):
        model_path = os.path.join(pretrained_model_path, "mp_rank_00_model_states.pt")
        state_dict = torch.load(model_path, map_location="cpu")
    elif os.path.isfile(os.path.join(pretrained_model_path, "pytorch_model.bin")):
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")

    for load_key in load_keys:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
            break
        else:
            print(f"### Magic_141 3D Transformer: missing key: {load_key} in the checkpoint: {model_path}.")
    _state_dict  = modify_state_dict(state_dict)["module"] # TODO DEBUG
    if not len(_state_dict.keys()) == 0:
        state_dict = _state_dict
    return state_dict



def load(module: nn.Module, state_dict, prefix="", strict=True, metadata=None):
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix,
                                         local_metadata=local_metadata,
                                         strict=strict, 
                                         missing_keys=missing_keys,
                                         unexpected_keys=unexpected_keys,
                                         error_msgs=error_msgs,
                                         )
    assert len(error_msgs) == 0, f"Meeting Error \n {error_msgs}"

    for name, child in module._modules.items():
        if child is not None:
            ms, us = load(child, prefix + name + ".")
            missing_keys = missing_keys + ms
            unexpected_keys = unexpected_keys + us
    return missing_keys, unexpected_keys

def load_state_dict(module: nn.Module, state_dict, strict=True):
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    return load(module, state_dict, prefix="", strict=strict, metadata=metadata)

def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward

def load(module: nn.Module, state_dict, prefix="", strict=True, metadata=None):
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix,
                                         local_metadata=local_metadata,
                                         strict=strict, 
                                         missing_keys=missing_keys,
                                         unexpected_keys=unexpected_keys,
                                         error_msgs=error_msgs,
                                         )
    assert len(error_msgs) == 0, f"Meeting Error \n {error_msgs}"

    for name, child in module._modules.items():
        if child is not None:
            ms, us = load(child, state_dict, prefix + name + ".", 
                          strict=strict, metadata=metadata)
            missing_keys = missing_keys + ms
            unexpected_keys = unexpected_keys + us
    return missing_keys, unexpected_keys

def load_state_dict(module: nn.Module, state_dict, strict=True):
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    return load(module, state_dict, prefix="", strict=strict, metadata=metadata)

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        audio_enable: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        # Audio Part args
        audio_block: nn.Module = None,
        audio_feature_all: Optional[torch.Tensor] = None,
        bsz: int = None,
        ot: int = None,
        face_mask_attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)
        
        # import pdb;pdb.set_trace()

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        
        # import pdb;pdb.set_trace()
        # Apply RoPE if needed. 
        # Every inference time add this again!
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        # print(f"Running Double Block {img_q.shape=} {txt_q.shape=}")

        # import pdb;pdb.set_trace()
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        
        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        # Cross Attention with audio feature
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )
        # import pdb; pdb.set_trace()
        # Audio Part
        if audio_block is not None:
            real_img = audio_block(audio_feature_all, img.view(bsz, ot, -1, 3072))
            img = img + real_img.view(bsz, -1, 3072) * face_mask_attn #face_mask
            # print("Double Block","Img Std:", img.std().item(), "Audio Std:", real_img.std().item())
        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        # Audio Part args
        audio_block: nn.Module = None,
        audio_feature_all: Optional[torch.Tensor] = None,
        bsz: int = None,
        ot: int = None,
        face_mask_attn: Optional[torch.Tensor] = None,
        img_shape_1: int = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        # print(f"Running Single Block")
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        output = apply_gate(output, gate=mod_gate)
        x = x + output
        
        if audio_block is not None:
            tmp_img, tmp_txt = x[:, :img_shape_1], x[:, img_shape_1:]
            real_img = tmp_img.view(bsz, ot, -1, 3072)
            real_img = audio_block(audio_feature_all, real_img)
            tmp_img = tmp_img + real_img.view(bsz, -1, 3072) * face_mask_attn
            x = torch.cat([tmp_img, tmp_txt], dim=1)
            # print("Single Block","Img Std:", tmp_img.std().item(), "Audio Std:", real_img.std().item())
        return x

class Magic141VideoDiffusionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    """
    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        rope_theta: int = 256,
        audio_enable: bool = False,
        use_face_mask: bool = False,
        audio_feature_dim: int = 768,
        audio_double_layer_skip: int = 1,
        audio_single_layer_skip: int = 1,
        audio_img_pos_embed: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.audio_guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )
        self.text_guidance_in = (            
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for i in range(mm_double_blocks_depth) # 20
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for i in range(mm_single_blocks_depth) # 40
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )


        self.gradient_checkpointing = False
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
        
    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length, height, width]

        if isinstance(self.config.patch_size, int):
            assert all(s % self.config.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.config.patch_size for s in latents_size]
        elif isinstance(self.config.patch_size, list):
            assert all(
                s % self.config.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.config.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.config.hidden_size // self.config.heads_num
        rope_dim_list = self.config.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.config.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def prepare_other_embedding(self, prompt_path, 
                                prompt_2_path,
                                neg_prompt_path=None,
                                neg_prompt_2_path=None,
                                prompt_mask_path=None,
                                neg_prompt_mask_path=None,
                                ):
        # Prepare for virtual prompt embedding
        self.prompt_embedding = torch.load(prompt_path)
        self.prompt_2_embedding = torch.load(prompt_2_path)
        self.prompt_mask = torch.load(prompt_mask_path)
        
        if neg_prompt_path is not None:
            self.neg_prompt_embedding = torch.load(neg_prompt_path)
        if neg_prompt_2_path is not None:
            self.neg_prompt_2_embedding = torch.load(neg_prompt_2_path)
        if neg_prompt_mask_path is not None:
            self.neg_prompt_mask = torch.load(neg_prompt_mask_path)

    def re_init_patch_embed(self, new_param_init="zero", audio_new_param_init="zero"):
        pass
        tmp_weights = self.img_in.proj.weight.clone()
        tmp_bias = self.img_in.proj.bias.clone()
        self.img_in.proj = torch.nn.Conv3d(
            2 * self.config.in_channels + 1, tmp_weights.shape[0], 
            kernel_size=self.config.patch_size, 
            stride=self.config.patch_size, bias=True,
        )
        with torch.no_grad():
            self.img_in.proj.weight[:, :self.config.in_channels] = tmp_weights # original weights
            if new_param_init == "zero":
                self.img_in.proj.weight[:, self.config.in_channels:] = \
                    torch.zeros(self.img_in.proj.weight[:, self.config.in_channels:].shape) # new weights initialized to zero
            elif new_param_init == "norm":
                torch.nn.init.normal_(self.img_in.proj.weight[:, self.config.in_channels:], 
                                      mean=tmp_weights.mean().item(), 
                                      std=tmp_weights.std().item())
            self.img_in.proj.bias = torch.nn.Parameter(tmp_bias)
        
                
    @classmethod
    def from_pretrained(cls, pretrained_model_path, subfolder=None, from_scratch=False, use_sst_state_load=False, model_additional_kwargs={}):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded loaded Magic141 3D Transformer from {pretrained_model_path} ...")

        config_file = "configs/model/video_dit_config.json"
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
                    
        prompt_paths = model_additional_kwargs.pop("prompt_paths")
        new_param_init = model_additional_kwargs.pop("new_param_init", "zero")
        audio_new_param_init = model_additional_kwargs.pop("audio_new_param_init", "zero")

        # model = cls.from_config(config, **model_additional_kwargs)
        config.update(model_additional_kwargs)
        model = cls(**config)

        # TODO: check what's this for?
        if from_scratch:
            model.re_init_patch_embed(new_param_init, audio_new_param_init)
            model.prepare_other_embedding(**prompt_paths)
            return model

        load_key = "module"

        if os.path.isdir(pretrained_model_path):
            model_path = os.path.join(pretrained_model_path, "mp_rank_00_model_states.pt")
        else:
            model_path = pretrained_model_path

        if use_sst_state_load:
            model.re_init_patch_embed(new_param_init, audio_new_param_init)
            model.prepare_other_embedding(**prompt_paths)
            state_dict = sst_load_state_dict(model_path)
            m, u = model.load_state_dict(state_dict, strict=False)

        else:
            state_dict = torch.load(model_path, mmap=True, map_location="cpu")

            if load_key in state_dict:
                state_dict = state_dict[load_key]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )
            m, u = model.load_state_dict(state_dict, strict=False)
            print(f"### Magic141 3D Transformer: missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

            model.re_init_patch_embed(new_param_init, audio_new_param_init)
            model.prepare_other_embedding(**prompt_paths)
        return model

    @classmethod
    def from_config_only(cls, model_additional_kwargs):
        config_file = "/mnt/weka/pretrained_weights/Magic141-video-t2v-720p/transformers/config.json"
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        prompt_paths = model_additional_kwargs.pop("prompt_paths")
        new_param_init = model_additional_kwargs.pop("new_param_init", "zero")
        audio_new_param_init = model_additional_kwargs.pop("audio_new_param_init", "zero")

        # model = cls.from_config(config, **model_additional_kwargs)
        config.update(model_additional_kwargs)
        model = cls(**config)
        model.re_init_patch_embed(new_param_init, audio_new_param_init)
        model.prepare_other_embedding(**prompt_paths)
        return model

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()    

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        # x: torch.Tensor,
        # t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        audio_feature: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
        txt_uncond: Optional[torch.Tensor] = None,
        audio_guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        text_guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        if_saining_loss: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        '''
         wav2vec_embeds: 1, 3072, 21, 1:  batch, frames, window_size=10, block=13, dim=768
        x = rearrange(x, 'b (n d) f l -> b f n d l', n=4, d=768) # l = 13

        '''

        
        B, C, video_length, H, W = hidden_states.shape
        video_length = hidden_states.size(2)
        ref_condition_mask = torch.zeros((B, 1, video_length, H, W), device=hidden_states.device, dtype=hidden_states.dtype)
        ref_condition_mask[:, :, :encoder_hidden_states.size(2), ...] = 1.
        encoder_hidden_states = torch.cat([encoder_hidden_states, \
            torch.zeros_like(encoder_hidden_states[:, :, :1, ...]).repeat(1, 1, video_length - encoder_hidden_states.size(2), 1, 1)], dim=2) \
            if video_length > 1 else encoder_hidden_states
        hidden_states = torch.cat([hidden_states, encoder_hidden_states, ref_condition_mask], dim=1)
        encoder_hidden_states = None

        # prepare rope position embedding
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(video_length, H, W)

        # prepare face_mask_attn feature
        face_mask_attn = 1.

        if face_mask is not None:
            with torch.no_grad():
                face_mask_attn = torch.nn.functional.interpolate(rearrange(face_mask, "b c f h w -> (b f) c h w"), (H // self.config.patch_size[1], W // self.config.patch_size[2]), mode="nearest")
                face_mask_attn = rearrange(face_mask_attn, "(b f) c h w -> b (f h w) c", f=video_length)
        
        # prepare x, t, text_embedding which is used in bellow
        x = hidden_states
        t = timestep
        # If haven't provided text_states, it will be default positive prompt embed
        # If have provided text_states, and batchsize of it is 1, repeat to same batchsize as hidden_states
        if text_states is None:
            text_states = self.prompt_embedding.repeat(hidden_states.shape[0], 1, 1).to(device=hidden_states.device, dtype=hidden_states.dtype)
            text_mask = self.prompt_mask.repeat(hidden_states.shape[0], 1).to(device=hidden_states.device)
            text_states_2 = self.prompt_2_embedding.repeat(hidden_states.shape[0], 1).to(device=hidden_states.device, dtype=hidden_states.dtype)
        elif text_states.size(0) == 1:
            text_states = text_states.repeat(hidden_states.shape[0], 1, 1).to(device=hidden_states.device, dtype=hidden_states.dtype)
            text_mask = text_mask.repeat(hidden_states.shape[0], 1).to(device=hidden_states.device)
            text_states_2 = text_states_2.repeat(hidden_states.shape[0], 1).to(device=hidden_states.device, dtype=hidden_states.dtype)
        # After upper, select only text embed as uncond
        if txt_uncond is not None:
            txt_embed_len = self.neg_prompt_embedding.size(1)
            text_states[txt_uncond, -txt_embed_len:] = self.neg_prompt_embedding.to(device=hidden_states.device, dtype=hidden_states.dtype)
            text_mask[txt_uncond, -txt_embed_len:] = self.neg_prompt_mask.to(device=text_mask.device, dtype=text_mask.dtype)
            text_states_2[txt_uncond] = self.neg_prompt_2_embedding.to(device=hidden_states.device, dtype=hidden_states.dtype)

        ## TODO: Guidance is already distilled in the model. [text embedding is ""]
        # guidance = torch.tensor(
        #                 [1] * B,
        #                 dtype=hidden_states.dtype,
        #                 device=hidden_states.device,
        #             ) * 1000.0

        if audio_guidance is not None:
            if type(audio_guidance) == (float) or type(audio_guidance) == (int):
                audio_guidance = torch.tensor(
                                [audio_guidance] * B,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )
        if text_guidance is not None:
            if type(text_guidance) == (float) or type(text_guidance) == (int):
                text_guidance = torch.tensor(
                                [text_guidance] * B,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                            )

        # Now, model start to forward
        out = {}
        img = x
        txt = text_states
        bsz, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        
        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            print("passing here to add guidance")
            if audio_guidance is None or text_guidance is None:
                raise ValueError(
                    "Didn't get text or audio guidance strength for guidance distilled model."
                )
            vec = vec + self.audio_guidance_in(audio_guidance) + self.text_guidance_in(text_guidance)

        # Embed image and text.
        img = self.img_in(img) # (B C F H W) flatten to (B C (F H W)) and transpose(B (F H W) C)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        # print(f"{img_seq_len=} {txt_seq_len=}")
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q
        
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for layer_num, block in enumerate(self.double_blocks):
            # import pdb;pdb.set_trace()
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]
            img, txt = block(*double_block_args)
            
        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        ori_x = x

        if len(self.single_blocks) > 0:
            for layer_num, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]
                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        # import pdb;pdb.set_trace()
        img = self.unpatchify(img, tt, th, tw)
        if if_saining_loss:
            return Transformer2DModelOutput(sample=img), ori_x, ori_audio_feature
        else:
            return Transformer2DModelOutput(sample=img)

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts




#################################################################################
#                             MAGIC_141_VIDEO_CONFIG Configs                              #
#################################################################################

MAGIC_141_VIDEO_CONFIG = {
    "Magic_141_Video-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "Magic_141_Video-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}



