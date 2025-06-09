import os

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

def chunk_comba_dplr_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = True
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p = p.transpose(1, 2)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        p = F.pad(p, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, p, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, p, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    p_beta = p * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, p_beta, decay, g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, p_beta, decay.unsqueeze(-1), g.unsqueeze(-1)]
    )
    decay = decay.squeeze(-1).cumsum(-1) # [B, H, n, c]
    decay_0 = decay - g.squeeze(-1) # [B, H, n, c]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    L_mask_0 = ((decay_0.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    # [B, H, n, c, d] @ [B, H, n, d, c] -> [B, H, n, c, c]
    attn = -((p_beta @ k.transpose(-1, -2)) * L_mask_0).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    M = attn
    # for U
    k_cumsum = attn @ v
    # for W
    k_cumdecay = attn @ (p_beta * decay_0[..., None].exp())
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    # decay = rearrange(decay, 'b h n c -> b h (n c)')
    # decay_0 = rearrange(decay_0, 'b h n c -> b h (n c)')
    # M = rearrange(M, 'b h n c d -> b h (n c) d')
    # w = rearrange(k_cumdecay, 'b h n c d -> b h (n c) d')
    # u = rearrange(k_cumsum, 'b h n c d -> b h (n c) d')
    if not head_first:
        o = o.transpose(1, 2)
        # decay = decay.transpose(1, 2)
        # decay_0 = decay_0.transpose(1, 2)
        # M = M.transpose(1, 2)
        # w = w.transpose(1, 2)
        # u = u.transpose(1, 2)
    return o, S



def chunk_comba_iplr_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = True
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p = p.transpose(1, 2)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        p = F.pad(p, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, p, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, p, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v_beta = v * beta[..., None]
    p_beta = p * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, v_beta, p_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, v_beta, p_beta, decay.unsqueeze(-1)]
    )
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    # for U
    attn = -((p_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v_beta

    attn = -((p_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumdecay = attn @ p_beta

    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    if not head_first:
        o = o.transpose(1, 2)
    return o, S