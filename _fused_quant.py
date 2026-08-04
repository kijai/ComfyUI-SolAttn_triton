"""One-pass INT8 quantization for BTHD tensors, with optional mean smoothing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_kernel(
    x_ptr, mean_ptr, xi_ptr, xs_ptr,
    T,
    s_b, s_t, s_h,  # x strides (last dim contiguous); xi/xs are dense allocations
    H: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    SUBTRACT_MEAN: tl.constexpr,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    valid = rows < T

    x = tl.load(
        x_ptr + batch * s_b + rows[:, None] * s_t + head * s_h + d[None, :],
        mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    if SUBTRACT_MEAN:
        m = tl.load(mean_ptr + batch_head * D + d).to(tl.float32)
        x = x - m[None, :]

    s = tl.max(tl.abs(x), axis=1) / 127.0
    s_safe = tl.where(s > 1e-8, s, 1e-8)
    xi = tl.extra.libdevice.round(x / s_safe[:, None])
    xi = tl.minimum(tl.maximum(xi, -127.0), 127.0).to(tl.int8)

    offs_out = ((batch * T + rows[:, None]) * H + head) * D + d[None, :]
    tl.store(xi_ptr + offs_out, xi, mask=valid[:, None])
    tl.store(xs_ptr + (batch * T + rows) * H + head, s_safe, mask=valid)


def quantize_bthd(x, mean=None, rows=16, num_warps=4):
    B, T, H, D = x.shape
    xi = torch.empty((B, T, H, D), device=x.device, dtype=torch.int8)
    xs = torch.empty((B, T, H), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(T, rows), B * H)
    _quant_kernel[grid](
        x, mean if mean is not None else x, xi, xs,
        T, x.stride(0), x.stride(1), x.stride(2), H, D, rows, mean is not None,
        num_warps=num_warps,
    )
    return xi, xs


__all__ = ["quantize_bthd"]
