"""Optional logging of Triton autotune sweeps.

Each new sequence length makes every autotuned kernel benchmark its configs,
stalling for seconds; without a log line that reads as a mysterious hang.
"""

import inspect
import logging
import time

import torch
import triton

_verbose = False


def _l2_sized_flush_buffer():
    """Triton benchmarks with a flat 256 MB buffer to evict L2 between runs.
    That lands on top of the kernel's own peak; the device's real L2 is enough."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    size = getattr(props, "L2_cache_size", 0) or (32 << 20)
    return torch.empty(size // 4, dtype=torch.int, device="cuda")


def lean_do_bench(fn, quantiles=None, **kwargs):
    """triton.testing.do_bench with the smaller flush buffer, patched only for
    the duration of this call so other extensions' autotuning is untouched."""
    driver = triton.runtime.driver.active
    original = driver.get_empty_cache_for_benchmark
    driver.get_empty_cache_for_benchmark = _l2_sized_flush_buffer
    try:
        return triton.testing.do_bench(fn, quantiles=quantiles, **kwargs)
    finally:
        driver.get_empty_cache_for_benchmark = original


def _supported_autotune_kwargs():
    """The optional autotune features this Triton actually accepts.

    ``cache_results`` is recent and ``do_bench`` older; passing either to a
    Triton that predates it raises TypeError while the module is still being
    imported, which takes the whole node down rather than losing one feature.
    """
    params = inspect.signature(triton.autotune).parameters
    extras = {}
    if "cache_results" in params:
        extras["cache_results"] = True       # persist timings across restarts
    if "do_bench" in params:
        extras["do_bench"] = lean_do_bench   # L2-sized flush buffer, not a flat 256 MB
    return extras


AUTOTUNE_EXTRAS = _supported_autotune_kwargs()


def set_verbose(enabled):
    global _verbose
    _verbose = bool(enabled)


def wrap(kernel, label):
    """Log when this autotuner resolves a new key (verbose mode only)."""
    original_run = kernel.run

    def run(*args, **kwargs):
        before = len(kernel.cache)
        # A fresh benchmark rebinds bench_time; a disk-cache hit leaves it alone.
        marker = getattr(kernel, "bench_time", None)
        start = time.perf_counter()
        result = original_run(*args, **kwargs)
        if _verbose and len(kernel.cache) > before:
            benched = getattr(kernel, "bench_time", None) is not marker
            how = "benchmarked" if benched else "read cached timings for"
            logging.info(
                f"[sol_attn] autotune: {label} {how} {len(kernel.configs)} "
                f"config(s) in {time.perf_counter() - start:.1f}s")
        return result

    kernel.run = run
    return kernel
