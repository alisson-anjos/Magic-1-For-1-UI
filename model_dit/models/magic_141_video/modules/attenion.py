import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

def get_ca_cu_seqlens(audio_len, img_len, batch_size):
    cu_seqlens = torch.zeros([batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cu_seqlens[i + 1] = (i + 1) * img_len
        
    cu_seqlens_q = cu_seqlens
    cu_seqlens = torch.zeros([batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cu_seqlens[i + 1] = (i + 1) * audio_len
        
    cu_seqlens_kv = cu_seqlens
    return cu_seqlens_q, cu_seqlens_kv

def get_ca_cu_seqlens_audio(audio_len, img_len):
    # import pdb;pdb.set_trace()
    batch_size =  audio_len
    cu_seqlens = torch.zeros([batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cu_seqlens[i + 1] = (i + 1) * img_len 
        cu_seqlens[i + 1] = cu_seqlens[i + 1] // audio_len
        
    cu_seqlens_q = cu_seqlens
    cu_seqlens = torch.zeros([batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cu_seqlens[i + 1] = (i + 1) * 1 #should be 1 for temporal audio length
        cu_seqlens[i + 1] = cu_seqlens[i + 1] 
        
    cu_seqlens_kv = cu_seqlens
    return cu_seqlens_q, cu_seqlens_kv

def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv
):  
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:,img_q_len:cu_seqlens_q[1]],
        joint_tensor_key=k[:,img_kv_len:cu_seqlens_kv[1]],
        joint_tensor_value=v[:,img_kv_len:cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    ### ! Warning: CP only works with Attention2.0.
    # local_rank = os.environ['LOCAL_RANK']
    # print(f"hybrid_seq_parallel_attn cost {time.perf_counter() - s} {local_rank=}")
    if img_kv_len != 0:
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["flash"]
        q = pre_attn_layout(q[:,cu_seqlens_q[1]:])
        k = pre_attn_layout(k[:,cu_seqlens_kv[1]:])
        v = pre_attn_layout(v[:,cu_seqlens_kv[1]:])
        max_seqlen_q = cu_seqlens_q[2] - cu_seqlens_q[1]
        max_seqlen_kv = max_seqlen_q
        attn2 = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q[1:] - cu_seqlens_q[1],
            cu_seqlens_kv[1:] - cu_seqlens_kv[1],
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        attn2 = attn2.view(
            1, max_seqlen_q, attn2.shape[-2], attn2.shape[-1]
        )  # reshape x to [b, s, a, d]
        attn2 = post_attn_layout(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
    else:
        attn = attn1
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
