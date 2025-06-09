# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import safe_exp, safe_exp2
from fla.utils import check_shared_mem


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'BK', 'BC', 'HEAD_FIRST', 'USE_OFFSETS'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_prepare_wy_repr_kernel_chunk32(
    k,
    g,
    beta,
    Aw,
    Au,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_Aw = tl.zeros([BC, BC], dtype=tl.float32)
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_beta = tl.load(p_beta, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_Aw += tl.dot(b_kb, tl.trans(b_k))

    b_Aw = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_Aw, 0)

    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,))
    b_Au = b_Aw * safe_exp(b_g[:, None] - b_g[None, :])

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_aw = tl.sum(tl.where(mask[:, None], b_Aw, 0), 0)
        b_au = tl.sum(tl.where(mask[:, None], b_Au, 0), 0)
        b_aw = b_aw + tl.sum(b_aw[:, None] * b_Aw, 0) * (tl.arange(0, BC) < i)
        b_au = b_au + tl.sum(b_au[:, None] * b_Au, 0) * (tl.arange(0, BC) < i)
        b_Aw = tl.where(mask[:, None], b_aw, b_Aw)
        b_Au = tl.where(mask[:, None], b_au, b_Au)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_Aw += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_Au += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    if HEAD_FIRST:
        p_Aw = tl.make_block_ptr(Aw + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au = tl.make_block_ptr(Au + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    else:
        p_Aw = tl.make_block_ptr(Aw + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_Au = tl.make_block_ptr(Au + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    tl.store(p_Aw, b_Aw.to(p_Aw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Au, b_Au.to(p_Au.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'BK', 'BC', 'USE_OFFSETS', 'HEAD_FIRST'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_prepare_wy_repr_kernel_chunk64(
    k,
    p,
    g,
    g0,
    beta,
    M,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_M1 = tl.zeros([BC, BC], dtype=tl.float32)
    b_M2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_M3 = tl.zeros([BC, BC], dtype=tl.float32)
    if HEAD_FIRST:
        pass
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_beta2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))
        p_g0 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_g0_2 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BC,), (0,))
        p_g1_2 = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT + BC,), (BC,), (0,))

    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))
    b_g0_1 = tl.load(p_g0, boundary_check=(0,))
    b_g0_2 = tl.load(p_g0_2, boundary_check=(0,))
    b_g1_1 = tl.load(p_g1, boundary_check=(0,))
    b_g1_2 = tl.load(p_g1_2, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            pass
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
            p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_p2 = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_pb = (b_p * b_beta[:, None]).to(b_p.dtype)
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_p2 = tl.load(p_p2, boundary_check=(0, 1))
        b_pb2 = (b_p2 * b_beta2[:, None]).to(b_p2.dtype)
        b_M1 += tl.dot(b_pb, tl.trans(b_k))
        b_M2 += tl.dot(b_pb2, tl.trans(b_k2))
        b_M3 += tl.dot(b_pb2, tl.trans(b_k))

    b_M1 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_M1, 0)
    b_M2 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_M2, 0)

    mask_c = tl.arange(0, BC)[:, None] >= tl.arange(0, BC)[None, :]
    mask_g = i_t * BT + tl.arange(0, BC) < T
    mask_g2 = i_t * BT + BC + tl.arange(0, BC) < T

    b_M1 = tl.where(mask_g[None, :] & mask_c, b_M1 * safe_exp2(b_g0_1[:, None] - b_g1_1[None, :]), 0)
    b_M2 = tl.where(mask_g2[None, :] & mask_c, b_M2 * safe_exp2(b_g0_2[:, None] - b_g1_2[None, :]), 0)
    b_M3 = tl.where(mask_g[None, :], b_M3 * safe_exp2(b_g0_2[:, None] - b_g1_1[None, :]), 0)

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_m1 = tl.sum(tl.where(mask[:, None], b_M1, 0), 0)
        b_m2 = tl.sum(tl.where(mask[:, None], b_M2, 0), 0)
        b_m1 = b_m1 + tl.sum(b_m1[:, None] * b_M1, 0) * (tl.arange(0, BC) < i)
        b_m2 = b_m2 + tl.sum(b_m2[:, None] * b_M2, 0) * (tl.arange(0, BC) < i)
        b_M1 = tl.where(mask[:, None], b_m1, b_M1)
        b_M2 = tl.where(mask[:, None], b_m2, b_M2)
    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_M1 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_M2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    # improve precision by disallowing tf32.
    b_M3 = -tl.dot(tl.dot(b_M2, b_M3, allow_tf32=False), b_M1, allow_tf32=False)

    if HEAD_FIRST:
        pass
    else:
        p_M1 = tl.make_block_ptr(M + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_M2 = tl.make_block_ptr(M + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_M3 = tl.make_block_ptr(M + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_M4 = tl.make_block_ptr(M + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))

    tl.store(p_M1, b_M1.to(p_M1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_M2, b_M2.to(p_M2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_M3, b_M3.to(p_M3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_M4, tl.zeros([BC, BC], dtype=tl.float32).to(p_M4.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'HEAD_FIRST', 'USE_OFFSETS'],
)
@triton.jit(do_not_specialize=['T'])
def fwd_recompute_w_u_kernel(
    k,
    v,
    g0,
    beta,
    w,
    u,
    M,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if HEAD_FIRST:
        pass
    else:
        p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_g0 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_M = tl.make_block_ptr(M + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g0 = tl.load(p_g0, boundary_check=(0,))
    b_M = tl.load(p_M, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_M, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            pass
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * tl.exp(b_g0[:, None])).to(b_k.dtype)
        b_w = tl.dot(b_M, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def fwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    g0: torch.Tensor,
    beta: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(BT, 32)
    BK = min(triton.next_power_of_2(K), 64)
    # bf16 should be good enough.
    M = torch.empty(B, *((H, T) if head_first else (T, H)), BT, device=k.device, dtype=k.dtype)

    fwd_fn = fwd_prepare_wy_repr_kernel_chunk64
    fwd_fn[(NT, B*H)](
        k=k,
        p=p,
        g=g,
        g0=g0,
        beta=beta,
        M=M,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BC=BC,
        HEAD_FIRST=head_first
    )
    w, u = fwd_recompute_w_u(
        k=p,
        v=v,
        g0=g0,
        beta=beta,
        M=M,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=chunk_size
    )
    return w, u, M


def fwd_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    g0: torch.Tensor,
    beta: torch.Tensor,
    M: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    fwd_recompute_w_u_kernel[(NT, B*H)](
        k=k,
        v=v,
        g0=g0,
        beta=beta,
        w=w,
        u=u,
        M=M,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return w, u


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'HEAD_FIRST', 'USE_OFFSETS']
)
@triton.jit(do_not_specialize=['T'])
def bwd_prepare_wy_repr_kernel(
    k,
    v,
    p,
    g0,
    g,
    beta,
    M,
    dw,
    du,
    dk,
    dv,
    dp,
    dbeta,
    dg0,
    dg,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dg0 = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    if HEAD_FIRST:
        pass
    else:
        p_beta = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_g0 = tl.make_block_ptr(g0 + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(M + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g0 = tl.load(p_g0, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            pass
        else:
            p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dp = tl.make_block_ptr(dp + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_p_beta_g0 = (b_p * b_beta[:, None] * tl.exp(b_g0[:, None])).to(b_p.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_p_beta_g0), allow_tf32=False)
        b_dp_beta_g0 = tl.dot(b_A, b_dw, allow_tf32=False)

        b_dp = b_dp_beta_g0 * b_beta[:, None] * tl.exp(b_g0[:, None])
        b_dbeta += tl.sum(b_dp_beta_g0 * b_p * tl.exp(b_g0[:, None]), 1)
        b_dg0 += tl.sum(b_dp * b_p, 1)
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))

    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA, 0).to(k.dtype.element_ty)

    b_dA2 = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            pass
        else:
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA2 += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dA2 = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA2, 0)
    b_dA2 = tl.dot(b_dA2.to(b_A.dtype), b_A)
    b_dA2 = tl.dot(b_A, b_dA2.to(b_A.dtype))
    b_dA2 = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA2, 0).to(k.dtype.element_ty)
    if HEAD_FIRST:
        p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_dA2 *= safe_exp2(b_g0[:, None] - b_g[None, :])
    b_dA *= safe_exp2(b_g0[:, None] - b_g[None, :])
    b_dA += b_dA2
    b_dA = b_dA.to(k.dtype.element_ty)
    # initial A to store single Diag(beta)PK
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            pass
        else:
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dp = tl.make_block_ptr(dp + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))
        # single kbeta
        b_p_beta = (b_p * b_beta[:, None]).to(b_p.dtype)
        # A = Diag(beta)PK
        b_A += tl.dot(b_p_beta, tl.trans(b_k))
        # two of d (Diag(beta)p)
        b_dp_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        # two of dbeta
        b_dbeta += tl.sum(b_dp_beta * b_p, 1)
        # two of dk in K
        b_dk = tl.dot(tl.trans(b_dA), b_p_beta, allow_tf32=False)
        # two of dk in betaP
        b_dp += b_dp_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
    # dA2 = d(gi-gj) = d (exp(gi-gj) * Diag(beta)PK) * Diag(beta)PK * exp(gi-gj) = dA2 * A
    b_dA2 = b_dA * b_A
    b_dg0 += tl.sum(b_dA2, axis=1)
    b_dg = - tl.sum(b_dA2, axis=0)
    if HEAD_FIRST:
        pass
    else:
        p_dg = tl.make_block_ptr(dg + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dg0 = tl.make_block_ptr(dg0 + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dbeta = tl.make_block_ptr(dbeta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dg0, b_dg0.to(p_dg0.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))

def bwd_prepare_wy_repr(
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g0: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    M: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(triton.next_power_of_2(K), CONST_TILING)
    BV = min(triton.next_power_of_2(V), CONST_TILING)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dp = torch.empty_like(k)
    dbeta = torch.empty_like(beta)
    dg = torch.empty_like(g)
    dg0 = torch.empty_like(g0)
    bwd_prepare_wy_repr_kernel[(NT, B * H)](
        k=k,
        v=v,
        p=p,
        g0=g0,
        g=g,
        beta=beta,
        M=M,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dp=dp,
        dbeta=dbeta,
        dg0=dg0,
        dg=dg,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first
    )
    return dk, dv, dp, dbeta, dg0, dg
