# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
from einops import rearrange
from fla.ops.comba.comba_iplr.fused_recurrent import RecurrentCombaIPLRFunction
from fla.ops.comba.comba_dplr.fused_recurrent import RecurrentCombaDPLRFunction

def fused_recurrent_comba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    head_first: bool = False,
    type: str = "iplr",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p is None:
        p = k
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
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if head_first:
        q, k, v, g, beta = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, g, beta))
    if type == "iplr":
        o, final_state = RecurrentCombaIPLRFunction.apply(
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
            use_qk_l2norm_in_kernel
        )
    elif type == "dplr":
        o, final_state = RecurrentCombaDPLRFunction.apply(
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
            use_qk_l2norm_in_kernel
        )
    else:
        raise ValueError(f"Only support iplr and dplr, unsupported type: {type}")
    if head_first:
        o = rearrange(o, 'b t h v -> b h t v')
    return o, final_state
