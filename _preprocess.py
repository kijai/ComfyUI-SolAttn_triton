"""Block summaries and routing thresholds shared by both forward kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._autotune_log import wrap as _wrap_autotune


BLOCK_SIZE = 64
# kc/vc are zero-padded to this many blocks so group tiles load unmasked;
# covers every GROUP the forward kernels autotune over.
GROUP_PAD = 64


def tau_vector(tau, heads, device):
    """Broadcast tau to the per-head vector the kernels index."""
    return torch.full((heads,), float(tau), device=device, dtype=torch.float32)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=2)
        for warps in (4, 8)
    ],
    key=["T"],
)
@triton.jit
def _reduce_kc_kernel(
    k_ptr,
    kc,
    T,
    s_b, s_t, s_h,  # k strides; last dim contiguous
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0) / block_len
    tl.store(
        kc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=2)
        for warps in (4, 8)
    ],
    key=["T"],
)
@triton.jit
def _reduce_vc_kernel(
    v_ptr,
    vc,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        v_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0)
    tl.store(
        vc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["T"],
)
@triton.jit
def _diag_threshold_kernel(
    q_ptr,
    kc_mean,
    kc_var_diag,
    kc_k3,
    kc_k4,
    global_threshold,
    softmax_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
    tau_ptr,
    CORNISH_FISHER: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    d_offsets = tl.arange(0, TILE_D)
    valid_d = d_offsets < D
    q_rows = q_start + tl.arange(0, BLOCK)
    q_values = tl.load(
        q_ptr + batch * s_b + q_rows[:, None].to(tl.int64) * s_t + head * s_h + d_offsets[None, :],
        mask=(q_rows < T)[:, None] & valid_d[None, :],
        other=0.0,
    )
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + d_offsets, mask=valid_d, other=0.0)
    var_kc = tl.load(kc_var_diag + batch_head * D + d_offsets, mask=valid_d, other=0.0)

    TAU = tl.load(tau_ptr + head)
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc, axis=0
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)

    offset = TAU
    if CORNISH_FISHER:
        # Cumulants add under the diagonal independence assumption; the
        # standardised skew/kurtosis are scale free so log2 cancels.
        k3_d = tl.load(kc_k3 + batch_head * D + d_offsets, mask=valid_d, other=0.0)
        k4_d = tl.load(kc_k4 + batch_head * D + d_offsets, mask=valid_d, other=0.0)
        q2 = q_centroid * q_centroid
        raw_var = tl.sum(q2 * var_kc, axis=0)
        raw_sd = tl.sqrt(tl.maximum(raw_var, 0.0) + 1.0e-12)
        g1 = tl.sum(q2 * q_centroid * k3_d, axis=0) / (raw_sd * raw_sd * raw_sd)
        g2 = tl.sum(q2 * q2 * k4_d, axis=0) / (raw_sd * raw_sd * raw_sd * raw_sd)
        g1 = tl.minimum(tl.maximum(g1, -2.0), 2.0)
        g2 = tl.minimum(tl.maximum(g2, -5.0), 5.0)
        z = TAU
        offset = (z
                  + (z * z - 1.0) * g1 / 6.0
                  + (z * z * z - 3.0 * z) * g2 / 24.0
                  - (2.0 * z * z * z - 5.0 * z) * g1 * g1 / 36.0)
        # Asymptotic expansion; keep within one sigma of the Gaussian answer.
        offset = tl.minimum(tl.maximum(offset, z - 1.0), z + 1.0)

    tl.store(
        global_threshold + (batch * N + q_block) * H + head,
        mean + offset * std,
    )


_wrap_autotune(_reduce_kc_kernel, "kc reduction")
_wrap_autotune(_reduce_vc_kernel, "vc reduction")
_wrap_autotune(_diag_threshold_kernel, "routing threshold")


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, padded, heads, head_dim = k.shape
    tokens = padded if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    blocks_padded = ((blocks + GROUP_PAD - 1) // GROUP_PAD) * GROUP_PAD
    kc = torch.zeros(
        (batch, blocks_padded, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.zeros_like(kc)
    grid = (triton.cdiv(head_dim, tile_d), blocks, batch * heads)
    _reduce_kc_kernel[grid](
        k,
        kc,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    _reduce_vc_kernel[grid](
        v,
        vc,
        tokens,
        v.stride(0), v.stride(1), v.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
    cornish_fisher: bool = False,
) -> torch.Tensor:
    batch, padded, heads, head_dim = q.shape
    tokens = padded if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))

    # kc is small enough that the moments are cheaper in torch than a kernel.
    kv = kc[:, :blocks].float().permute(0, 2, 1, 3)            # [B,H,blocks,D]
    kc_mean = kv.mean(dim=2).contiguous()
    centred = kv - kc_mean.unsqueeze(2)
    kc_var_diag = centred.pow(2).mean(dim=2).contiguous()
    if cornish_fisher:
        kc_k3 = centred.pow(3).mean(dim=2).contiguous()
        kc_k4 = (centred.pow(4).mean(dim=2) - 3.0 * kc_var_diag.pow(2)).contiguous()
    else:
        kc_k3 = kc_var_diag
        kc_k4 = kc_var_diag

    global_threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _diag_threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var_diag,
        kc_k3,
        kc_k4,
        global_threshold,
        scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        tau_vector(tau, heads, q.device),
        cornish_fisher,
    )
    return global_threshold


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
    cornish_fisher: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (kc, vc, threshold). ``tokens`` is the true sequence length."""
    kc, vc = _reduce_kv(k, v, tokens)
    threshold = _compute_diag_threshold(q, kc, tau=tau, scale=scale, tokens=tokens,
                                        cornish_fisher=cornish_fisher)
    return kc, vc, threshold


__all__ = ["prepare"]
