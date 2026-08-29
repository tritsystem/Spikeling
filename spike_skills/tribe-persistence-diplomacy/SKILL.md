---
name: tribe-persistence-diplomacy
description: Use when working on Tribe's save/offline-catchup, alliances, proximity trade, or seasons systems.
keywords: tribe_persist, world_tribe.gd, trade_envoy, offline catch-up, TribeTradeUI, proximity trade
---

# Tribe persistence & diplomacy

Godot 4 trust/Spikeling-brain NPC sim, public at
github.com/tritsystem/tribe (own scoped repo, separate from the
profile-root repo -- don't confuse the two when pushing).

Real systems: save + offline-catch-up ("leave for a week" and the world
advances coherently), alliances, proximity-based trade, and a
deliberate no-forced-winner + seasons design (the game doesn't railroad
toward one tribe "winning").

Trade specifically runs through physical courier NPCs
(`trade_envoy.gd`), not an abstract instant exchange -- the player sends
offers with the `'` key and gets a real accept/decline panel
(`TribeTradeUI`) for incoming requests. This was built via the agent
pipeline then independently re-verified headless -- don't assume agent-
built Tribe systems are unverified just because they weren't hand-
written.
