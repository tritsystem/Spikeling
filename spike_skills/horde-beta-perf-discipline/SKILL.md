---
name: horde-beta-perf-discipline
description: Use when working in this repo (horde-beta-version-1) on team_id/animation/weapon-path logic or per-frame target-scanning.
keywords: horde-beta, horde-defense-beta, team_id bug, weapon-path, target-scan
---

# Horde-beta perf & bug discipline

Real, already-fixed bugs in this exact repo: team_id, animation, and
weapon-path issues. Also a real O(n^2) per-frame target-scan
performance bug, profiled and fixed for roughly a 17x improvement --
if target-scanning logic changes again, re-profile rather than assuming
the old complexity class is still fine after the edit.

Cross-substrate note: this repo is also where Spikeling first got wired
into a game (horde-defense-beta), a first for that integration
direction -- Tribe had used Spikeling-brain NPCs already, but this was
the first time in the reverse direction of "bring Spikeling into an
existing game project" rather than building the game around it.
