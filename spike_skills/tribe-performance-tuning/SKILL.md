---
name: tribe-performance-tuning
description: Use when Tribe (or any Godot project) needs performance work -- profiling before changing anything.
keywords: tribe performance, physics-frame, hive brain, MultiMesh trees, LOD active-body
---

# Tribe performance discipline

The real, hard-won lesson: profile physics-frame ms FIRST, always.
Guessed wrong 4 times before profiling actually found the real cause at
Epic-scale lag.

What actually fixed it, in the real order found: LOD active-body count,
a hive-brain pattern (shared computation instead of per-NPC), LOW
graphics preset specifically for shadows-off, MultiMesh for trees
(instancing instead of individual nodes), and opaque water (transparency
was a real, non-obvious cost). None of these were the first guess --
that's the point: don't reach for a "likely" optimization without a
profile confirming it's the actual bottleneck.
