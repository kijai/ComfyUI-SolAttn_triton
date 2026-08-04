"""Optional logging of Triton autotune sweeps.

Each new sequence length makes every autotuned kernel benchmark its configs,
stalling for seconds; without a log line that reads as a mysterious hang.
"""

import logging
import time

_verbose = False


def set_verbose(enabled):
    global _verbose
    _verbose = bool(enabled)


def wrap(kernel, label):
    """Log when this autotuner benchmarks a new key (verbose mode only)."""
    original_run = kernel.run

    def run(*args, **kwargs):
        before = len(kernel.cache)
        start = time.perf_counter()
        result = original_run(*args, **kwargs)
        if _verbose and len(kernel.cache) > before:
            logging.info(
                f"[sol_attn] autotune: {label} benchmarked {len(kernel.configs)} "
                f"config(s) in {time.perf_counter() - start:.1f}s ")
        return result

    kernel.run = run
    return kernel
