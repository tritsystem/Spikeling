---
name: noise-cancelling-frontend
description: Use when working on real 2-mic structured-noise cancellation experiments (the noise_cancelling_frontend finding).
keywords: noise-cancelling frontend, structured noise, C922 mic, spectral overlap
---

# Noise-cancelling frontend finding

First PHYSICAL test (real 2-mic C922, not just simulation) of the
reservoir "structured noise nulled at readout" sim result. Confirmed
and replicated twice, but only after catching two real bugs along the
way: a verdict-printer that lied about its own results, and a spectral-
overlap confound. The real, load-bearing condition for this to work:
selective cancellation only succeeds when the target signal is
spectrally separated from the interferer -- don't expect this to work
if the two signals overlap in frequency.
