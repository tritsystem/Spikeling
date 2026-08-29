---
name: review-discipline
description: Use when Spike's Reviewer specialist fires, or when the user asks to peer-review, audit, or check a change for overclaiming. Read-only review discipline -- ported from spiking_orchestrator.py's real Reviewer specialist, not written from scratch.
---

# Review discipline

Ported directly from Spike's own working Reviewer specialist
(`spiking_orchestrator.py::_reviewer_task()` and the `REVIEW_TOOLS`
comment in `voice_commands.py`), not invented for this skill file. This
is the real discipline that specialist already runs under -- this file
just makes it independently readable/editable without touching Python.

## Ground rule: read-only, on purpose

The Reviewer gets `Read`, `Grep`, `Glob` only -- no `Edit`, no `Write`,
no shell. This is deliberate: a reviewer that can edit the code can also
edit it to make its own claim pass. The tool restriction IS the
integrity mechanism, not a permissions afterthought.

## What to actually do

1. Peer-review the change for the stated task. Check it against the real
   files, not against what the change claims about itself.
2. If specialists were spawned mid-task (dynamic specialists), review
   their work specifically too -- not just the original change. Firing
   order alone doesn't tell you what changed; check explicitly.
3. **Call out overclaiming.** This is the one instruction the real
   specialist prompt states explicitly, verbatim -- a review that just
   confirms the work "looks fine" without checking claims against files
   isn't a review.

## Severity, and what happens with it

The real Reviewer's output feeds a ternary gate
(`spiking_orchestrator.py`'s `REVIEW_GATE_HI` / `REVIEW_GATE_LO`,
0.65 / 0.15): certain issues correct automatically, certain-clean skips
correction, the ambiguous middle band decides on a >=0.5 severity
read. Strong-issue language ("wrong", "fails", "incorrect", "breaks",
"bug", "broken", "crash") pushes severity up. Write review findings in
terms a downstream severity read can act on -- state plainly whether
something is actually broken, not just stylistically different from how
you'd have done it.
