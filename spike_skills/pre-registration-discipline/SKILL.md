---
name: pre-registration-discipline
description: Use when Spike's PreRegister specialist fires, or when the user asks to pre-register a claim before making a change. Falsifiable-claim discipline -- ported from voice_commands.py's real PREREGISTER_PREAMBLE and spiking_orchestrator.py's PreRegister framing, not written from scratch.
---

# Pre-registration discipline

Ported directly from Spike's own working PreRegister specialist
(`voice_commands.py::PREREGISTER_PREAMBLE` and the
`"PreRegister": f"Before any edit, state ONE falsifiable claim..."`
framing in `spiking_orchestrator.py::_agent_task()`), not invented for
this skill file. This is the gbranaa-hue method's own core discipline --
pre-register a falsifiable prediction before doing the work -- applied
to code changes specifically, and this file just makes that real,
working preamble independently readable/editable without touching
Python.

## Why this exists, specifically

"Did it work?" gets checked against a claim stated BEFORE the change,
not judged after the fact by whoever did the work. That ordering is the
entire point -- a prediction written after seeing the result isn't a
prediction, it's a description, and it can't be wrong in a way that
would ever get caught.

## What to actually do

Before writing any code, state **ONE** short, concrete, falsifiable
claim about what the change will do once implemented -- something a
reviewer could check against the real files afterward. For example:
"adds function X to file Y that returns Z", or "command A now triggers
on phrase B and calls C".

- Not a plan. Not a list of steps. **One checkable sentence.**
- Reply with only that sentence, nothing else -- no preamble, no hedging,
  no "I will now implement..." framing around it.
- If the claim can't be stated as one falsifiable sentence, the task is
  probably not scoped tightly enough yet -- that's a signal to narrow it,
  not to write a vaguer claim.

## How it gets used downstream

The real pipeline treats this as the anchor for review: a separate,
independent peer-review pass (see the `review-discipline` skill) checks
the pre-registered claim against the actual current files, not against
the implementer's own summary of what it did. A claim that can't be
checked this way -- too vague, too broad, or stated as an intention
rather than an observable outcome -- defeats the whole mechanism before
review even starts.
