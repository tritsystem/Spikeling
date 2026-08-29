---
name: topological-phononics-ssh
description: Use when working on topological-phononics' SSH reservoir, defect-tolerance experiments, or Fibonacci/quasicrystal connectivity research.
keywords: SSH reservoir, topological-phononics, defect-tolerance, chiral symmetry, Duffing model
---

# Topological phononics (SSH reservoir)

Published, public at github.com/gbranaa4-hue/topological-phononics.
Core finding, fully scoped: SSH reservoir defect-tolerance holds ONLY
when chiral symmetry holds, and REVERSES in a physical Duffing model --
don't generalize "topological protection helps" without checking which
symmetry regime and which physical model is actually in play.

Fibonacci/quasicrystal connectivity sub-thread: a 3-parameter recursive-
rule reservoir vs. O(N^2)-stored random connectivity, confound-hunted
across 9 scripts (a real construction-artifact bug was found and fixed
via verified prefix-nesting along the way). Final honest cost, smoothly
scale-characterized N=50-10000: ~1.4-2x for linear recall, ~1-17% for
NARMA10 -- a real but modest efficiency win, not reported as bigger than
it is.
