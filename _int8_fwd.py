"""Sol-Attn with an INT8 exact branch (Sage-style smoothed K, per-token scales).

K smoothing shifts every score in a row by the same constant, so softmax and
routing are unchanged and bit-identical to the BF16 kernel. The approximate
branch stays BF16: the pooled tensors are O(N*d), so quantizing them saves
nothing.
"""

import torch
import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ModuleNotFoundError:
    TensorDescriptor = None

from ._fused_prep import fused_preprocess
from ._tri_fwd import _has_tma, _is_gfx1151


BLOCK = 64
GROUP = 32


def _pad_to_blocks(q, k, v, block):
    """Pad tokens to whole blocks for the TMA path's unmasked descriptor I/O."""
    tokens = q.shape[1]
    blocks = (tokens + block - 1) // block
    padded = blocks * block
    if padded == tokens:
        return q, k, v, tokens, padded
    pad = (0, 0, 0, 0, 0, padded - tokens)
    return (torch.nn.functional.pad(q, pad), torch.nn.functional.pad(k, pad),
            torch.nn.functional.pad(v, pad), tokens, padded)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _forward_int8(
    q_desc, v_desc, kc_desc, vc_desc, o_desc,
    qi_ptr, qs_ptr, ki_ptr, ks_ptr,
    threshold,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    TP,  # padded token count: the batch stride of qi/qs/ki/ks (T is only the mask bound)
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    q_start = q_block * BLOCK_SIZE

    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK_SIZE, D])
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    # INT8 query tile + per-token scales, staged once for the whole program.
    q_rows = q_start + token_offsets
    q_valid = q_rows < T
    qi = tl.load(
        qi_ptr + ((batch * TP + q_rows[:, None]) * H + head) * D + d_offsets[None, :],
        mask=q_valid[:, None],
        other=0,
    )
    qs = tl.load(qs_ptr + (batch * TP + q_rows) * H + head, mask=q_valid, other=0.0)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(threshold + (batch * NT + q_block) * H + head)
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load([batch, group_start, head, 0]).reshape([GROUP_SIZE, D])
        vc = vc_desc.load([batch, group_start, head, v_tile * BV]).reshape([GROUP_SIZE, BV])

        # --- routing + approximate correction, BF16 ---
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(approximate[None, :], scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(approximate_probability.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NT - 1, tail_length, BLOCK_SIZE).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(approximate_probability * lengths[None, :], axis=1)
        row_max = new_max

        # --- exact branch, INT8 QK ---
        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP_SIZE, exact_offsets)
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k_valid = k_rows < T

            ki = tl.load(
                ki_ptr + ((batch * TP + k_rows[:, None]) * H + head) * D + d_offsets[None, :],
                mask=k_valid[:, None],
                other=0,
            )
            ks = tl.load(ks_ptr + (batch * TP + k_rows) * H + head, mask=k_valid, other=0.0)

            acc = tl.dot(qi, ki.T, out_dtype=tl.int32)
            exact_scores = acc.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2
            exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))

            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = v_desc.load([batch, kv_start, head, v_tile * BV]).reshape([BLOCK_SIZE, BV])
            output = output * alpha[:, None] + tl.dot(exact_probability.to(v.dtype), v)
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


@triton.jit
def _forward_int8_ptr(
    q_ptr, v_ptr, kc_ptr, vc_ptr, o_ptr,
    qi_ptr, qs_ptr, ki_ptr, ks_ptr,
    threshold,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    T,
    TP,    # padded token count: batch stride of o/qi/qs/ki/ks (our allocations)
    NPAD,  # padded block count: batch stride of kc/vc
    sq_b, sq_t, sq_h,   # q/v strides (last dim must be contiguous)
    sv_b, sv_t, sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Same math as _forward_int8. q/v/o mask their ragged token tails; kc/vc
    # are GROUP-padded allocations and load unmasked.
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    bv_offsets = v_tile * BV + tl.arange(0, BV)
    q_start = q_block * BLOCK_SIZE

    q_rows_ok = q_start + token_offsets < TP
    q = tl.load(
        q_ptr + batch * sq_b + (q_start + token_offsets[:, None]) * sq_t
        + head * sq_h + d_offsets[None, :],
        mask=q_rows_ok[:, None],
        other=0.0,
    )
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    # INT8 query tile + per-token scales, staged once for the whole program.
    q_rows = q_start + token_offsets
    q_valid = q_rows < T
    qi = tl.load(
        qi_ptr + ((batch * TP + q_rows[:, None]) * H + head) * D + d_offsets[None, :],
        mask=q_valid[:, None],
        other=0,
    )
    qs = tl.load(qs_ptr + (batch * TP + q_rows) * H + head, mask=q_valid, other=0.0)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(threshold + (batch * NT + q_block) * H + head)
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = tl.load(
            kc_ptr + ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + d_offsets[None, :]
        )
        vc = tl.load(
            vc_ptr + ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + bv_offsets[None, :]
        )

        # --- routing + approximate correction, BF16 ---
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        routed = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_kv
        ) & valid
        exact = tl.where(q_in_sink, valid, routed)

        approximate = valid & ~exact
        approximate_scores = tl.where(approximate[None, :], scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(approximate_probability.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NT - 1, tail_length, BLOCK_SIZE).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(approximate_probability * lengths[None, :], axis=1)
        row_max = new_max

        # --- exact branch, INT8 QK ---
        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP_SIZE, exact_offsets)
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k_valid = k_rows < T

            ki = tl.load(
                ki_ptr + ((batch * TP + k_rows[:, None]) * H + head) * D + d_offsets[None, :],
                mask=k_valid[:, None],
                other=0,
            )
            ks = tl.load(ks_ptr + (batch * TP + k_rows) * H + head, mask=k_valid, other=0.0)

            acc = tl.dot(qi, ki.T, out_dtype=tl.int32)
            exact_scores = acc.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2
            exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))

            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = tl.load(
                v_ptr + batch * sv_b + (kv_start + token_offsets[:, None]) * sv_t
                + head * sv_h + bv_offsets[None, :],
                mask=k_valid[:, None],
                other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(exact_probability.to(v.dtype), v)
            row_max = new_max

    tl.store(
        o_ptr + ((batch * TP + q_start + token_offsets[:, None]) * H + head) * D
        + bv_offsets[None, :],
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_rows_ok[:, None],
    )


_forward_int8_ptr_autotuned = triton.autotune(
    configs=[
        # Kept small: every config costs seconds of compile per new T.
        triton.Config({"BV": 128, "GROUP_SIZE": 64}, num_warps=4, num_stages=1),
        triton.Config({"BV": 128, "GROUP_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"BV": 128, "GROUP_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BV": 128, "GROUP_SIZE": 32}, num_warps=4, num_stages=1),
        triton.Config({"BV": 64, "GROUP_SIZE": 64}, num_warps=4, num_stages=1),
    ],
    key=["T"],
)(_forward_int8_ptr)


def sol_attn_int8(q, k, v, *, scale=None, tau=1.0, sink_blocks=(0, 0), sink_q=(0, 0),
                  cornish_fisher=False):
    """Sol-Attn with an INT8 exact branch. Same contract as the BF16 kernel."""
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, _, heads, head_dim = q.shape
    use_tma = _has_tma(q.device)
    if use_tma:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k, v, tokens, padded = _pad_to_blocks(q, k, v, BLOCK)
        if TensorDescriptor is None:
            raise RuntimeError("TensorDescriptor is unavailable for the TMA path.")
    else:
        # Pointer kernel and quantizer take strides; skip the copies.
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        tokens = padded = q.shape[1]
    blocks = triton.cdiv(tokens, BLOCK)

    kc, vc, threshold, qi, qs, ki, ks = fused_preprocess(
        q, k, v, tau=tau, scale=scale, tokens=tokens,
        cornish_fisher=cornish_fisher
    )

    output = torch.empty((batch, padded, heads, head_dim),
                         device=v.device, dtype=v.dtype)
    if not use_tma:
        args = (
            q, v, kc, vc, output,
            qi, qs, ki, ks,
            threshold,
            scale,
            int(sink_blocks[0]),
            int(sink_blocks[1]),
            int(sink_q[0]),
            int(sink_q[1]),
            tokens,
            padded,
            kc.shape[1],
            q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
        )
        meta = dict(H=heads, D=head_dim, NT=blocks, BLOCK_SIZE=BLOCK)
        if _is_gfx1151(q.device):
            _forward_int8_ptr[(1, blocks, batch * heads)](
                *args, **meta,
                BV=128, GROUP_SIZE=32,
                num_warps=4, num_stages=1,
            )
        else:
            grid = lambda META: (head_dim // META["BV"], blocks, batch * heads)
            _forward_int8_ptr_autotuned[grid](*args, **meta)
        return output[:, :tokens]
    block_shape = [1, BLOCK, 1, head_dim]
    summary_shape = [1, GROUP, 1, head_dim]
    _forward_int8[(1, blocks, batch * heads)](
        TensorDescriptor.from_tensor(q, block_shape),
        TensorDescriptor.from_tensor(v, block_shape),
        TensorDescriptor.from_tensor(kc, summary_shape),
        TensorDescriptor.from_tensor(vc, summary_shape),
        TensorDescriptor.from_tensor(output, block_shape),
        qi, qs, ki, ks,
        threshold,
        scale,
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
        tokens,
        padded,
        heads,
        head_dim,
        blocks,
        head_dim,
        BLOCK,
        GROUP,
    )
    return output[:, :tokens]


__all__ = ["sol_attn_int8"]
