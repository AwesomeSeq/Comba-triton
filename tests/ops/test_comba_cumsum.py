# -*- coding: utf-8 -*-

import os

import pytest
import torch

from fla.ops.comba.chunk import chunk_comba_cumsum_scalar_fwd
from fla.ops.utils.testing import assert_close
from fla.utils import device, device_platform

compiled_mode = os.getenv("COMPILER_MODE") == "1"
if compiled_mode:
    test_b_list = [1]
    test_t_list = [64]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [1, 15, 63, 300]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [32, 64, 100, 256]
test_h_list = [2]


def rev_cumsum(s, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(s, dims=[dim]), dim), dims=[dim])


def cumsum_local_reference(s, reverse=False, head_first=False, chunk_size=128):
    o_0 = torch.zeros_like(s)
    o_1 = torch.zeros_like(s)
    T = s.size(2) if head_first else s.size(1)
    fn = torch.cumsum if not reverse else rev_cumsum
    for i in range(0, T, chunk_size):
        if head_first:
            assert NotImplementedError
        else:
            s_chunk = s[:, i:i+chunk_size]
            o_1[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o_1)
            o_0[:, i:i+chunk_size] = o_1[:, i:i+chunk_size] - s_chunk

    return o_0, o_1


def cumsum_global_reference(s, reverse=False, head_first=False):
    fn = torch.cumsum if not reverse else rev_cumsum
    return fn(s.float(), dim=2).to(s) if head_first else fn(s.float(), dim=1).to(s)




@pytest.mark.parametrize("B", [32])
@pytest.mark.parametrize("T", [256, 1024, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
def test_cumsum_local_scalar(B, T, H, dtype, head_first, reverse, chunk_size):
    if head_first:
        s = torch.randn((B, H, T), dtype=dtype, device=device).requires_grad_()
    else:
        s = torch.randn((B, T, H), dtype=dtype, device=device).requires_grad_()
    ref_0, ref_1 = cumsum_local_reference(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    tri_0, tri_1 = chunk_comba_cumsum_scalar_fwd(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref_0, tri_0, 0.001 if dtype == torch.float else 0.003)
    assert_close("local cumsum scalar", ref_1, tri_1, 0.001 if dtype == torch.float else 0.003)