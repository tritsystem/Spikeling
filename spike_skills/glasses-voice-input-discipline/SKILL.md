---
name: glasses-voice-input-discipline
description: Use when a message may have arrived via voice/glasses (glasses_hook.py, in the Spikeling repo) and could be terse or garbled.
keywords: glasses_hook.py, voice/glasses input, terse or garbled
---

# Glasses voice input discipline

Messages may arrive via voice/glasses through `glasses_hook.py` (in the
Spikeling repo) -- text from this path can arrive terse or garbled
compared to typed input. Confirm intent before big or risky actions
rather than guessing at unclear phrasing when a message looks like it
could be a mis-transcription rather than a deliberate short instruction.
This is a standing rule, not a one-off caution -- it applies any time
input plausibly came through this channel, not just when it's stated
explicitly.
