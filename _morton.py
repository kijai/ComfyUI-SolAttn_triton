"""Morton (Z-order) token reordering for Wan, to make 64-token blocks compact in 3D.

Video latents are laid out frame-major then row-major, so a 64-token attention
block is a thin 2 x 32 strip of one frame. Attention in a video DiT is locally
3D, so that strip touches many blocks that each carry little mass. Z-ordering
makes each block a roughly 4 x 4 x 4 neighbourhood, which concentrates the mass
into fewer blocks and lets the same quality be reached at higher sparsity.

The permutation is applied once before the block stack and inverted after it,
with the RoPE frequencies permuted to match, so it is mathematically a no-op
for dense attention.

IMPORTANT: the permute-or-not decision is made in a single place -- the block-0
pre-hook, which sees the hidden states and the RoPE table together. An earlier
version decided for the RoPE inside rope_encode and for the tokens in the hook,
against separate guards; when those disagreed (reference_latent prepends tokens,
context_latents appends them) the positions were permuted while the tokens were
not, and nothing undid it. That corrupts the output completely rather than
degrading it. Never split this decision.

Everything is gated on transformer_options["sol_morton"], so the hooks are
inert unless the patch node put that flag there.
"""

import logging
import math

import torch


_PERM_CACHE = {}
_INSTALLED = set()


def morton_perm(grid, device, curve="3d"):
    """Z-order permutation and its inverse.

    curve="3d"       interleave t/h/w equally.
    curve="2d_frame" Z-order within each frame, frames left in original order.
                     Use this when the temporal axis is not uniformly spaced --
                     MiniMax-H3's FRAME_PER_TOKEN is (1, 4, 4, 4, 4), so
                     index-adjacent frames are 1 or 4 real frames apart and a 3D
                     curve groups temporally distant tokens together.
    """
    key = (tuple(int(x) for x in grid), curve)
    hit = _PERM_CACHE.get(key)
    if hit is None:
        frames, height, width = key[0]
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        rem = linear - z * area
        y = rem // width
        x = rem - y * width

        def part1by2(value):
            value = value & 0x1FFFFF
            value = (value | (value << 32)) & 0x1F00000000FFFF
            value = (value | (value << 16)) & 0x1F0000FF0000FF
            value = (value | (value << 8)) & 0x100F00F00F00F00F
            value = (value | (value << 4)) & 0x10C30C30C30C30C3
            value = (value | (value << 2)) & 0x1249249249249249
            return value

        if curve == "2d_frame":
            # frame index stays the most significant key, so frames never mix
            code = (z << 42) | part1by2(x) | (part1by2(y) << 1)
        else:
            code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        _PERM_CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)


_logged = set()


def _log_once(message):
    if message not in _logged:
        _logged.add(message)
        logging.info(f"[sol_attn] Wan Morton: {message}")


def install_wan_morton(diffusion_model):
    """Idempotently install the reorder hooks. No effect without the flag."""
    if id(diffusion_model) in _INSTALLED:
        return
    if not hasattr(diffusion_model, "blocks"):
        raise RuntimeError("Morton reordering needs .blocks on the diffusion model")

    model = diffusion_model
    blocks = model.blocks
    first, last = blocks[0], blocks[-1]

    def _decide(x, freqs, transformer_options):
        """(perm, inverse, permuted_freqs) or None. The single decision point."""
        grid = transformer_options.get("grid_sizes")
        if grid is None:
            _log_once("transformer_options has no grid_sizes; inactive")
            return None
        tokens = int(math.prod(int(g) for g in grid))
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[1] != tokens:
            _log_once(f"token count {tuple(x.shape)} != video grid {tuple(grid)} "
                      f"({tokens}); reference/context streams present, inactive")
            return None
        if not torch.is_tensor(freqs) or freqs.shape[1] != tokens:
            _log_once(f"rope table {tuple(freqs.shape)} does not match the video grid "
                      f"({tokens}); inactive")
            return None
        curve = transformer_options.get("sol_morton_curve", "3d")
        perm, inverse = morton_perm(grid, x.device, curve)
        return perm, inverse, freqs.index_select(1, perm)

    def pre_hook(module, args, kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        if not transformer_options.get("sol_morton") or not args:
            return None

        if module is first:
            model._sol_morton_state = None
            decision = _decide(args[0], kwargs.get("freqs"), transformer_options)
            if decision is None:
                return None
            perm, inverse, freqs = decision
            model._sol_morton_state = (perm, inverse, freqs)
            kwargs = {**kwargs, "freqs": freqs}
            return (args[0].index_select(1, perm),) + tuple(args[1:]), kwargs

        # Later blocks reuse the permuted rope table computed once above; the
        # hidden states are already in Morton order and simply flow through.
        state = getattr(model, "_sol_morton_state", None)
        if state is None:
            return None
        kwargs = {**kwargs, "freqs": state[2]}
        return args, kwargs

    def post_hook(_module, _args, _kwargs, output):
        state = getattr(model, "_sol_morton_state", None)
        if state is None:
            return None
        model._sol_morton_state = None
        _perm, inverse, _freqs = state
        if not torch.is_tensor(output) or output.shape[1] != inverse.numel():
            logging.error("[sol_attn] Wan Morton: block stack changed the token count "
                          f"({tuple(output.shape)}); cannot restore order")
            return None
        return output.index_select(1, inverse)

    for block in blocks:
        block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    last.register_forward_hook(post_hook, with_kwargs=True)
    _INSTALLED.add(id(diffusion_model))


__all__ = ["morton_perm", "install_wan_morton"]
