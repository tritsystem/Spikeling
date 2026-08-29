---
name: spikeling-runtime-core
description: Use when working on Spikeling's core LIF/STDP runtime (core/runtime/runtime.py, pyspike.py) or .spk pipeline files.
keywords: STDP, pyspike.py, core/runtime/runtime.py, LTD, LTP, plasticity
---

# Spikeling runtime core

Real, load-bearing fact: STDP exists ONLY in `core/runtime/runtime.py`,
NOT in `pyspike.py` (confirmed by direct grep -- zero STDP hooks there).
Don't assume plasticity is available just because a `.spk` file runs;
check which engine is actually executing it.

Documented STDP gotchas, found by testing, not assumed:
- STDP relaxes toward baseline instead of growing under asymmetric
  firing rates -- it doesn't monotonically strengthen just because two
  neurons co-fire often.
- Simultaneous-fire (dt = 0) lands in the LTD (weakening) branch, not
  LTP. A dt-of-zero case is easy to miss in test coverage and will
  silently do the opposite of what's expected.

Known gap: the LIF-cascade orchestrator (`spiking_orchestrator.py`) has
a promotion/structural-learning path (dynamic specialists get promoted
into the fixed roster) but no demotion path -- a promoted specialist
that stops being useful stays promoted.
