---
name: tribe-direct-voice
description: Use when working on Tribe's direct_voice.gd deterministic dialogue system or tribe_llm.gd's say_as_direct path.
keywords: direct_voice.gd, DirectVoice, PHRASE_BANK, trust_band, spikes-to-text
---

# Tribe DirectVoice

Deterministic, LLM-free "spikes-to-text" voice decoder --
`DirectVoice.compose_line(personality, trust, betrayed_count,
recall_confidence, described_memory)`. Real trust bands via
`trust_band()`: hostile / wary / neutral / warming / trusting, each
personality x band combination pulled from a real `PHRASE_BANK` (5
personalities x 5 bands).

This exists ALONGSIDE the Ollama-backed `say_as()` path in
`tribe_llm.gd`, not as a replacement for it -- `say_as_direct()` was
added, `say_as()` was left untouched. Verified 12/12 real checks both
before and after the later turtle-island removal, so it's a stable,
independently-tested piece even though the rest of the file changed
around it.
