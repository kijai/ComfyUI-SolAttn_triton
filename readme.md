## Experimental Sol-Attn implementation for ComfyUI

work in progress

https://owen718.github.io/pubs/43-2026-arxiv-solattn/

Tested on 4090 and 5090 with MiniMax H3, and on Radeon 8060S (gfx1151) with
ROCm 7.15 and Triton 3.7. The gfx1151 path uses tuned pointer kernels; TMA
tensor descriptors remain NVIDIA-only. Triton compilation makes the first run slower.

Balance the quality/speed with start/end percent and tau.
