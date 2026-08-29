---
name: pc-compute-harnessing
description: Use when a task on this machine is compute-heavy (training, large simulation, batch inference) and could use the GPU.
keywords: RTX 5060, CUDA-capable PyTorch, GPU harnessing
---

# PC compute harnessing

This machine has an RTX 5060 with CUDA-capable PyTorch already
installed -- harness the GPU/CPU for compute-heavy work whenever viable
rather than defaulting to a slower CPU-only path or assuming no
accelerator is available. When speed and risk/quality trade off against
each other on this machine's work, the established default is to favor
quality, not the fastest shortcut.
