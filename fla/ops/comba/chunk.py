# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from fla.ops.comba.comba_iplr.chunk import ChunkCombaIPLRFunction
from fla.ops.comba.comba_dplr.chunk import ChunkCombaDPLRFunction


@torch.compiler.disable
def chunk_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    type: str = None
):
    if p is None:
        p = k
    assert q.dtype == k.dtype == v.dtype == p.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, H, T] if head_first=True, or [B, T, H] if head_first=False."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode"
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        beta, g = map(lambda x: rearrange(x, 'b h t -> b t h'), (beta, g))
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "Scale must be positive."

    if type == "iplr":
        o, final_state = ChunkCombaIPLRFunction.apply(
        q,
        k,
        v,
        p,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        False,
        use_qk_l2norm_in_kernel
    )
    elif type == "dplr":
        o, final_state = ChunkCombaDPLRFunction.apply(
            q,
            k,
            v,
            p,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            False,
            use_qk_l2norm_in_kernel
        )
    else:
        raise ValueError(f"Only support iplr and dplr, unsupported type: {type}")
    if head_first:
        o = rearrange(o, 'b t h v -> b h t v')
    return o, final_state
