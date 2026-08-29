---
name: corrector-discipline
description: Use when Spike's Corrector specialist fires, or when the user asks to fix exactly what a review flagged. Bounded, scoped correction discipline -- ported from spiking_orchestrator.py's real ternary review gate and _corrector_task(), and voice_commands.py's real one-shot correction pass, not written from scratch.
---

# Corrector discipline

Ported directly from Spike's own working Corrector specialist
(`spiking_orchestrator.py::_corrector_task()`, the real `gate_review()`
ternary severity gate that decides whether Corrector fires at all, and
`voice_commands.py`'s real "Cog 5" one-shot correction pass), not
invented for this skill file.

## Ground rule: fix exactly what review flagged, nothing broader

The real task framing is deliberately narrow: `"Fix exactly what review
flagged for: {task}"`. Not "improve the code," not "also clean up
anything else you notice while you're in there" -- a correction pass
that wanders scope stops being checkable against the review that
triggered it, and reintroduces the exact overclaiming problem review
exists to catch.

If dynamically-spawned specialists did work mid-task, factor their work
into the fix too -- not just the original change -- but the review
findings are still the boundary of what gets touched.

## Bounded, on purpose -- not a retry loop

Correction fires **at most twice** in the pyspike orchestrator (`self.
_corrections < 2`), and the standalone voice-command pipeline runs
exactly **one** correction pass, period. This is deliberate, not a
missing feature: an unbounded correct-review-correct-review loop can
spiral, and a cap forces the honest outcome to surface --
`corrected_after_review` or `review_failed_uncorrected` -- rather than
hiding an unfixable issue behind indefinite retries.

## The gate that decides whether you fire at all

Review output is scored into a severity in [0, 1], not a flat keyword
hit/no-hit:
- Strong issue markers (wrong, fails, incorrect, breaks, bug, broken,
  crash): +0.30 each.
- Weak issue markers (missing, overclaim, issue, does not, doesn't,
  should): +0.15 each.
- Escalation markers (critical, major, security, data loss, regression,
  breaks the build): +0.35 each.
- De-escalation markers (minor, nit, nitpick, trivial, typo, style,
  cosmetic): -0.20 each.

Severity >= 0.65 is certain-issue (correct). Severity <= 0.15 is
certain-clean (skip). The middle band only corrects at severity >= 0.5.
A real bug this gate's own history caught: an earlier version only
checked escalation language INSIDE the ambiguous band, which let "Found
a critical security issue... this introduces a regression" score 0.15
from "issue" alone and land in hard_safe -- skipped, uncorrected. Fixed
by always scoring every marker together, not gating the escalation check
behind a band decision it should have been informing.

## Log the correction attempt either way

Whether or not the fix fully lands, the correction attempt gets logged
to the ledger honestly -- `corrected_after_review` on success,
`review_failed_uncorrected` when the correction itself comes back empty.
The ledger doesn't get to look cleaner than what actually happened.
