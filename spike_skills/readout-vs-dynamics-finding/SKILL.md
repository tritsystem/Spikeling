---
name: readout-vs-dynamics-finding
description: Use when debugging reservoir-computing dead channels or evaluating whether an observable-orbit mechanism transfers between reservoirs.
keywords: D4 reservoir, dead channels, observable-orbit, QRC
---

# Readout vs dynamics finding

A D4 reservoir's dead-channel problem splits into two genuinely
different causes: part is readout-fixable (an observation/measurement
problem -- the information is there but not being read out correctly)
and part is geometry-only (a real generation limitation, not fixable by
changing the readout). The QRC observable-orbit mechanism only
half-transfers between these -- don't assume a readout fix will resolve
a dead channel without checking which category it's actually in
first.
