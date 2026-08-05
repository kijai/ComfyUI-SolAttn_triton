"""Fused preprocess for the INT8 path.

The smoothing mean is derived from the pooled keys (no second K read), and the
smoothed pooled keys are exactly zero-mean, so the threshold reduces to
tau = beta * sigma. K is read twice, Q and V once each.
"""

import torch
import triton
import triton.language as tl

from ._preprocess import _reduce_kv, BLOCK_SIZE, tau_vector
from ._fused_quant import quantize_bthd, quantize_v_per_channel


@triton.jit
def _q_quant_threshold_kernel(
    q_ptr, kc_var_ptr, qi_ptr, qs_ptr, thr_ptr,
    softmax_scale,
    T,
    TP,  # padded token count: the batch stride of qi/qs (T is only the mask bound)
    s_b, s_t, s_h,  # q strides (last dim contiguous)
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    tau_ptr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)

    offs = ((batch * TP + rows[:, None]).to(tl.int64) * H + head) * D + d[None, :]
    x = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    # --- INT8 quantization, per token ---
    s = tl.max(tl.abs(x), axis=1) / 127.0
    s_safe = tl.where(s > 1e-8, s, 1e-8)
    xi = tl.minimum(tl.maximum(x / s_safe[:, None], -127.0), 127.0)
    xi = tl.where(xi >= 0, xi + 0.5, xi - 0.5).to(tl.int8)
    tl.store(qi_ptr + offs, xi, mask=valid[:, None])
    tl.store(qs_ptr + (batch * TP + rows) * H + head, s_safe, mask=valid)

    # --- routing threshold from the same load (zero-mean kc: tau = beta*sigma) ---
    centroid = tl.sum(x, axis=0) / q_len
    var_kc = tl.load(kc_var_ptr + batch_head * D + d)
    log2_scale = softmax_scale * 1.4426950408889634
    c2 = centroid * centroid
    variance = tl.sum(c2 * var_kc, axis=0) * (log2_scale * log2_scale)

    TAU = tl.load(tau_ptr + head)
    tl.store(thr_ptr + (batch * N + q_block) * H + head,
             TAU * tl.sqrt(variance + 1.0e-6))


def fused_preprocess(q, k, v, *, tau, scale, tokens=None, int8_pv=True):
    """Returns (kc, vc, threshold, qi, qs, ki, ks, vi, vsc) with K smoothed.

    ``tokens`` is the true sequence length; q may be padded past it (TMA path).
    """
    B, padded, H, D = q.shape
    T = padded if tokens is None else int(tokens)
    N = triton.cdiv(T, BLOCK_SIZE)

    kc, vc = _reduce_kv(k, v, T)

    # Centre the pooled keys with their own mean; only valid blocks take part.
    k_mean = kc[:, :N].mean(dim=1, dtype=torch.float32)            # [B,H,D]
    kc[:, :N] = (kc[:, :N].float() - k_mean.unsqueeze(1)).to(kc.dtype)
    centred = kc[:, :N].float().permute(0, 2, 1, 3)                # [B,H,N,D], zero mean
    kc_var = centred.pow(2).mean(dim=2).contiguous()               # [B,H,D]

    # k may be shorter than q here
    ki, ks = quantize_bthd(k, mean=k_mean.reshape(B * H, D).contiguous(), out_rows=padded)
    vi, vsc = (quantize_v_per_channel(v, out_rows=padded) if int8_pv else (None, None))

    # Quantize Q and compute thresholds from one load.
    qi = torch.empty((B, padded, H, D), device=q.device, dtype=torch.int8)
    qs = torch.empty((B, padded, H), device=q.device, dtype=torch.float32)
    threshold = torch.empty((B, N, H), device=q.device, dtype=torch.float32)
    _q_quant_threshold_kernel[(N, B * H)](
        q, kc_var, qi, qs, threshold, scale, T, padded,
        q.stride(0), q.stride(1), q.stride(2), H, N, D, BLOCK_SIZE,
        tau_vector(tau, H, q.device),
        num_warps=4,
    )
    return kc, vc, threshold, qi, qs, ki, ks, vi, vsc


__all__ = ["fused_preprocess"]
