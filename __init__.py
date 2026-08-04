"""Sol-Attn (arXiv 2607.24027) as a ComfyUI attention override.

The "Patch Sol-Attn" node installs an ``optimized_attention_override`` on the
model, keeping the patch per-model and giving it the sigma schedule for dense
warm-up steps. ``SOL_ATTN=1`` installs a global override for CLI benchmarks.
"""

import logging
import os
from functools import partial

import torch

from comfy.ldm.modules.attention import (
    attention_pytorch,
    register_attention_function,
    wrap_attn,
)
from comfy_api.latest import ComfyExtension, io

from ._autotune_log import set_verbose as _set_autotune_verbose

try:
    from ._tri_fwd import sol_attn as _sol_attn_kernel, _has_tma
    _IMPORT_ERROR = None
except Exception as exc:  # triton / torch version issues
    _sol_attn_kernel = None
    _has_tma = None
    _IMPORT_ERROR = exc

try:
    from ._int8_fwd import sol_attn_int8 as _sol_attn_int8_kernel
    _INT8_IMPORT_ERROR = None
except Exception as exc:
    _sol_attn_int8_kernel = None
    _INT8_IMPORT_ERROR = exc


HEAD_DIM = 128

_stats = {"sparse": 0, "dense_fallback": 0, "outside_range": 0, "errors": 0}
_seen = set()


def sol_attn_stats():
    """Dispatch counters since process start (or last reset)."""
    return dict(_stats)


def reset_sol_attn_stats():
    for key in _stats:
        _stats[key] = 0
    _seen.clear()


def _log_once(key, message):
    if key not in _seen:
        _seen.add(key)
        logging.info(f"[sol_attn] {message}")


def _log_kernel_failure(exc):
    # Full traceback on the first occurrence of each distinct failure; a short
    # line on repeats so a failing run stays diagnosable without spam.
    key = ("kernel_failure", type(exc).__name__, str(exc))
    first = key not in _seen
    _seen.add(key)
    logging.error(f"[sol_attn] kernel failed ({exc}); falling back", exc_info=first)


def _ineligible(q, k, mask, dim_head, min_tokens):
    """Why this call can't use Sol-Attn, or None if it can. q/k are BTHD."""
    if _sol_attn_kernel is None:
        return f"kernel import failed: {_IMPORT_ERROR}"
    if q.device.type != "cuda":
        return "not cuda"
    if q.dtype != torch.bfloat16:
        return f"dtype {q.dtype} (kernel is bf16-only)"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != 128"
    if mask is not None:
        return "masked attention"
    if q.shape[1] != k.shape[1]:
        return "cross-attention (kept dense)"
    if q.shape != k.shape:
        # GQA or any other q/k mismatch would silently index wrong.
        return f"q/k shape mismatch {tuple(q.shape)} vs {tuple(k.shape)}"
    if q.shape[1] < min_tokens:
        return f"seq {q.shape[1]} < {min_tokens}"
    return None


def _run(q, k, v, heads, skip_reshape, skip_output_reshape, scale,
         tau, min_tokens, verbose, int8_qk=False, sink_blocks=(0, 0),
         sink_q=(0, 0), use_tma=False):
    """Returns the attention output, or None if this call should stay dense."""
    if skip_reshape:
        b, _, _, dim_head = q.shape          # BHND
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    else:
        b, _, dim_head = q.shape             # B, N, heads*dim_head
        dim_head //= heads
        qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

    reason = _ineligible(qs, ks, None, dim_head, min_tokens)
    if reason is not None:
        _stats["dense_fallback"] += 1
        if verbose:
            _log_once((tuple(qs.shape), reason), f"dense {tuple(qs.shape)}: {reason}")
        return None

    # No contiguous() here: the kernels take strides, so H3's interleaved qkv
    # views go in without copies.
    kernel = _sol_attn_int8_kernel if int8_qk else _sol_attn_kernel
    out = kernel(
        qs, ks, vs,
        scale=scale, tau=tau, sink_blocks=sink_blocks, sink_q=sink_q,
        use_tma=use_tma,
    )  # BTHD
    _stats["sparse"] += 1
    if verbose:
        mode = "int8" if int8_qk else "bf16"
        # The kernels also require SM90+ and a Triton with TensorDescriptor, so
        # report the path actually taken rather than what was asked for.
        path = "tma" if (use_tma and _has_tma(qs.device)) else "pointer"
        _log_once((tuple(qs.shape), "sparse", mode, path),
                  f"sparse {tuple(qs.shape)} tau={tau} {mode} {path}")

    if skip_output_reshape:
        return out.transpose(1, 2)           # BHND
    return out.reshape(b, -1, heads * dim_head)


BLOCK_SIZE = 64


def _sink_blocks(transformer_options, tokens, mode):
    """(exact-KV blocks, dense-query blocks) for MiniMax-H3's conditioning rows.

    H3 packs [text][cond][ref][audio][video] into one sequence; sparsifying the
    conditioning rows costs sync and prompt adherence. exact_kv measures ~3%,
    exact_kv_and_rows ~17%, so exact-KV is the default and rows are opt-in.
    """
    if mode == "off":
        return (0, 0), (0, 0)
    span = (transformer_options or {}).get("sol_h3_video_span")
    if span is None:
        return (0, 0), (0, 0)
    video_start, video_stop = span
    if tokens < video_stop or video_start <= 0:
        return (0, 0), (0, 0)
    blocks = (0, (video_start + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return blocks, (blocks if mode == "exact_kv_and_rows" else (0, 0))


def make_override(tau=1.0, min_tokens=4096,
                  sigma_start=None, sigma_end=None, verbose=False,
                  int8_qk=False, sink_conditioning="exact_kv", use_tma=False,
                  previous=None):
    """Build an optimized_attention_override callable.

    ``previous`` chains any override already installed on the model: every path
    that declines hands off to it first, falling through to ``func`` only if
    there is none.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def dense():
            target = func if previous is None else partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None:
            _stats["dense_fallback"] += 1
            return dense()

        # Sampling-percentage gate, so the paper's dense warm-up steps work.
        if sigma_start is not None or sigma_end is not None:
            sigmas = kwargs.get("transformer_options", {}).get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                if (sigma_start is not None and sigma > sigma_start) or \
                   (sigma_end is not None and sigma < sigma_end):
                    _stats["outside_range"] += 1
                    return dense()

        tokens = q.shape[2] if skip_reshape else q.shape[1]
        sink, sink_q = _sink_blocks(kwargs.get("transformer_options"), tokens,
                                    sink_conditioning)
        if verbose and sink != (0, 0):
            _log_once((tokens, sink, sink_q),
                      f"conditioning sink: KV blocks {sink} exact, dense query blocks {sink_q}")

        try:
            out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                       kwargs.get("scale", None), tau,
                       min_tokens, verbose, int8_qk, sink, sink_q, use_tma)
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
            return dense()
        return dense() if out is None else out

    return override


def _compose_module_patch(module, patched_forward):
    """Gate an object-patched attention forward (e.g. KJNodes' mem-efficient
    Sage): calls Sol-Attn would take run the stock forward and reach the
    override; the rest keeps the patch. Gate params come from
    transformer_options["sol_compose"]; when absent the patch runs as-is.
    """
    stock = type(module).forward

    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        if not isinstance(options, dict):
            options = next((a for a in args if isinstance(a, dict) and "sol_compose" in a), {})
        gate = options.get("sol_compose")
        x = args[0] if args else None
        take = gate is not None and torch.is_tensor(x) and x.device.type == "cuda" \
            and x.dtype == torch.bfloat16 and x.ndim in (2, 3)
        if take:
            # H3 packs tokens first (s, dim); Wan/LTX2 are batch-first.
            tokens = x.shape[0] if x.ndim == 2 else x.shape[1]
            take = tokens >= gate["min_tokens"]
        if take:
            sigmas = options.get("sigmas")
            if sigmas is not None:
                sigma = float(sigmas[0])
                take = not (sigma > gate["sigma_start"] or sigma < gate["sigma_end"])
        if take:
            return stock(module, *args, **kwargs)
        return patched_forward(*args, **kwargs)

    forward._sol_composed = True
    return forward


_COMPOSE_HOOKED = set()


def _install_compose_hooks(model, attn_attr):
    """Compose at sampling time, once all object patches are applied: a node
    downstream of ours overwrites the same object-patch key, so execute-time
    composition alone loses. The pre-hooks re-wrap any foreign attn forward
    before each block runs; inert unless sol_compose is published.
    """
    if id(model) in _COMPOSE_HOOKED:
        return

    def pre_hook(block, args):
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        fwd = attn.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_sol_composed", False):
            return None
        if getattr(fwd, "__func__", None) is type(attn).forward:
            return None  # unpatch leaves the stock forward as an instance attr
        attn.forward = _compose_module_patch(attn, fwd)
        _log_once(("composed", attn_attr),
                  f"composing with a patched {attn_attr}.forward; Sol-Attn takes "
                  "eligible self-attention calls, the patch keeps the rest")
        return None

    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    _COMPOSE_HOOKED.add(id(model))


class SolAttnPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SolAttnPatch",
            display_name="Patch Sol-Attn",
            is_experimental=True,
            category="sol_attn",
            description="Sparsify self-attention with Sol-Attn (arXiv 2607.24027), with "
                        "optional Morton token reordering, INT8 QK, and an exact-KV sink "
                        "for MiniMax-H3's packed conditioning rows. bf16 + head_dim 128 "
                        "only; everything else falls back to the existing attention backend.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("tau", default=1.2, min=0.0, max=4.0, step=0.05,
                               tooltip="Threshold beta. Higher is sparser: 1.0 ~ 16% of "
                                       "blocks kept exact, 1.5 ~ 7%, 2.0 ~ 2.7%."),
                io.Float.Input("start_percent", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Run dense before this point. The paper uses 0.2."),
                io.Float.Input("end_percent", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_tokens", default=4096, min=0, max=1 << 20, step=512,
                             tooltip="Sequences shorter than this stay dense."),
                io.Boolean.Input("int8_qk", default=False,
                                 tooltip="INT8 QK in the exact branch (Sage-style: smoothed K, "
                                         "per-token scales). Measured free in quality; helps at "
                                         "tau<=1.5, a net loss at tau>=2.0 where the quantize "
                                         "pass outweighs the shrinking exact branch."),
                io.Combo.Input("sink_conditioning", options=["exact_kv", "exact_kv_and_rows", "off"],
                               default="exact_kv",
                               tooltip="MiniMax-H3 only. exact_kv: every query sees the packed "
                                       "text/audio/reference rows exactly (~3% cost). "
                                       "exact_kv_and_rows: also runs those query rows dense, making "
                                       "the generated audio stream exact (~20% cost). "
                                       "No effect on other models."),
                io.Boolean.Input("morton", default=False,
                                 tooltip="Reorder video tokens into Morton (Z-order) so each "
                                         "64-token block is a compact 3D neighbourhood instead of a "
                                         "2-row strip, which makes routing far more accurate at a "
                                         "given density. Exactly neutral for dense attention. "
                                         "Wan and MiniMax-H3 only; logged and skipped elsewhere."),
                io.Combo.Input("morton_curve", options=["3d", "2d_frame"], default="2d_frame",
                               tooltip="3d interleaves t/h/w equally. 2d_frame Z-orders within "
                                       "each frame and leaves frame order alone -- use it when the "
                                       "temporal axis is not uniformly spaced (MiniMax-H3's frame "
                                       "spacing is non-uniform; try this if 3d degrades at some "
                                       "frame counts)."),
                io.Boolean.Input("verbose", default=False),
                io.Boolean.Input("use_tma", default=False,
                                 tooltip="Use the TMA descriptor kernels instead of the "
                                         "pointer ones. They need contiguous, block-padded "
                                         "q/k/v, so enabling this copies the inputs: peak "
                                         "VRAM goes from ~1x (bf16) / ~2x (int8) a q/k/v "
                                         "tensor to ~4x, plus the copy bandwidth. Off by "
                                         "default because it has not measured faster on any "
                                         "tested GPU. Requires SM90+ and Triton 3.3+; "
                                         "ignored otherwise. 'verbose' logs the path used."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, tau, start_percent, end_percent,
                min_tokens, int8_qk, sink_conditioning, morton,
                morton_curve, verbose, use_tma=False) -> io.NodeOutput:
        if _sol_attn_kernel is None:
            raise RuntimeError(f"Sol-Attn kernel unavailable: {_IMPORT_ERROR}")
        if int8_qk and _sol_attn_int8_kernel is None:
            raise RuntimeError(f"Sol-Attn INT8 kernel unavailable: {_INT8_IMPORT_ERROR}")

        diffusion_model = model.get_model_object("diffusion_model")
        is_h3 = hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")
        is_wan = hasattr(diffusion_model, "rope_encode") and hasattr(diffusion_model, "blocks")

        # H3 publishes its segment layout from the same hooks Morton uses, so the
        # conditioning sink needs them installed even when reordering is off.
        reorder = False
        if is_h3 and (morton or sink_conditioning != "off"):
            from ._morton_h3 import install_h3_hooks

            install_h3_hooks(diffusion_model)
            reorder = morton
        elif is_wan and morton:
            from ._morton import install_wan_morton

            install_wan_morton(diffusion_model)
            reorder = True
        elif morton:
            logging.warning(
                f"[sol_attn] Morton reordering skipped: {type(diffusion_model).__name__} "
                "is neither Wan-style nor MiniMax-H3. Sol-Attn itself still applies."
            )

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        m = model.clone()
        previous = m.model_options["transformer_options"].get("optimized_attention_override")
        if previous is not None:
            logging.info("[sol_attn] chaining onto an existing attention override; "
                         "Sol-Attn takes first refusal and delegates everything else to it")

        # Forward-level patches bypass optimized_attention entirely, so there is
        # nothing to chain onto -- gate each one instead (see _compose_module_patch).
        composed = []
        for key, patched in list(m.object_patches.items()):
            if not key.endswith(".forward"):
                continue
            owner = key.rsplit(".", 2)[-2].lower()
            if "attn" not in owner or "cross" in owner or owner == "attn2":
                continue  # Sol-Attn never takes cross-attention; leave it patched
            module = m.get_model_object(key[: -len(".forward")])
            m.add_object_patch(key, _compose_module_patch(module, patched))
            composed.append(key)
        if composed:
            logging.info(
                f"[sol_attn] composed with {len(composed)} object-patched attention "
                f"forward(s) (e.g. {composed[0]}): Sol-Attn takes eligible "
                "self-attention calls, the existing patch keeps the rest")
        # Downstream nodes overwrite the same keys; the hooks catch those at sampling time.
        if is_h3:
            _install_compose_hooks(diffusion_model, "attn")
        elif is_wan:
            _install_compose_hooks(diffusion_model, "self_attn")

        m.model_options["transformer_options"]["sol_compose"] = {
            "sigma_start": sigma_start, "sigma_end": sigma_end,
            "min_tokens": min_tokens}
        m.model_options["transformer_options"]["optimized_attention_override"] = \
            make_override(tau=tau, min_tokens=min_tokens,
                          sigma_start=sigma_start, sigma_end=sigma_end,
                          verbose=verbose, int8_qk=int8_qk,
                          sink_conditioning=sink_conditioning,
                          use_tma=use_tma, previous=previous)
        if reorder:
            m.model_options["transformer_options"]["sol_morton"] = True
            m.model_options["transformer_options"]["sol_morton_curve"] = morton_curve
        _set_autotune_verbose(verbose)
        reset_sol_attn_stats()
        return io.NodeOutput(m)


@wrap_attn
def attention_sol(q, k, v, heads, mask=None, attn_precision=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Registry-visible entry point using defaults (no sigma gating)."""
    if mask is None:
        try:
            out = _run(q, k, v, heads, skip_reshape, skip_output_reshape,
                       kwargs.get("scale", None), 1.0, 4096, False)
            if out is not None:
                return out
        except Exception as exc:
            _stats["errors"] += 1
            _log_kernel_failure(exc)
    return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)


register_attention_function("sol_attn", attention_sol)

if os.environ.get("SOL_ATTN", "0") not in ("0", "", "false"):
    if _sol_attn_kernel is None:
        logging.error(f"[sol_attn] SOL_ATTN set but kernel import failed: {_IMPORT_ERROR}")
    else:
        import comfy.ldm.modules.attention as _attn_mod

        _attn_mod.optimized_attention = attention_sol
        _attn_mod.optimized_attention_masked = attention_sol
        logging.info("[sol_attn] global override active (node patch is preferred)")

class SolAttnExtension(ComfyExtension):
    async def get_node_list(self):
        return [SolAttnPatch]


async def comfy_entrypoint() -> SolAttnExtension:
    return SolAttnExtension()
