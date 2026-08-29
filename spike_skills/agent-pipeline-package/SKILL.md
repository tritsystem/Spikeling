---
name: agent-pipeline-package
description: Use when working on the universal agent-pipeline + settings-UI package built for distribution (e.g. to family/other users).
keywords: agent-pipeline package, settings UI, querySelector collision
---

# Agent pipeline package

Universal agent-pipeline + HTML settings UI, packaged for the user's
brother as a standalone deliverable (not just an internal tool). A real
bug was found here that's worth remembering as a class, not just a
one-off: a querySelector collision that LOOKED like a caching bug at
first glance but wasn't -- worth checking selector uniqueness before
assuming a stale-cache explanation for UI state that won't update.
