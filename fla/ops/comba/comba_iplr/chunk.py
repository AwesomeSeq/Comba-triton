# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.comba.comba_iplr.wy_fast import bwd_prepare_wy_repr, fwd_prepare_wy_repr, fwd_recompute_w_u
from fla.ops.utils import chunk_local_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    g = chunk_local_cumsum(g, chunk_size, offsets=offsets, indices=indices, head_first=head_first)
    # obtain WY representation. u is actually the new v.
    w, u, Aw, Au = fwd_prepare_wy_repr(
        k=k,
        v=v,
        p=p,
        beta=beta,
        g=g,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )

    # obtain output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return g, o, Aw, Au, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u = fwd_recompute_w_u(
        k=p,
        v=v,
        beta=beta,
        Aw=Aw,
        Au=Au,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        dh=None,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk2, dv, dp, db, dg2 = bwd_prepare_wy_repr(
        k=k,
        v=v,
        p=p,
        beta=beta,
        g=g,
        Aw=Aw,
        Au=Au,
        dw=dw,
        du=dv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=offsets, indices=indices, head_first=head_first)
    return dq, dk, dv, dp, db, dg, dh0


class ChunkCombaIPLRFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
        use_qk_l2norm_in_kernel: bool = False
    ):
        chunk_size = 64
        q_orig = q
        k_orig = k
        p_orig = p

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
            p = l2norm_fwd(p)

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        g, o, Aw, Au, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            p=p,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q_orig, k_orig, v, p_orig, g, beta, Aw, Au, initial_state, offsets, indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.head_first = head_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, p, g, beta, Aw, Au, initial_state, offsets, indices = ctx.saved_tensors
        if ctx.use_qk_l2norm_in_kernel:
            q, q_orig = l2norm_fwd(q), q
            k, k_orig = l2norm_fwd(k), k
            p, p_orig = l2norm_fwd(p), p
        dq, dk, dv, dp, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            p=p,
            g=g,
            beta=beta,
            Aw=Aw,
            Au=Au,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
            head_first=ctx.head_first,
            chunk_size=ctx.chunk_size
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_orig, dq)
            dk = l2norm_bwd(k_orig, dk)
            dp = l2norm_bwd(p_orig, dp)
        return dq.to(q), dk.to(k), dv.to(v), dp.to(p), dg.to(g), db.to(beta), None, dh0, None, None, None, None
