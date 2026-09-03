# SNN agent-routing vs fixed pipeline — measured token cost

Repo under test: **doorcam** (github.com/tritsystem/doorcam). Model: claude-haiku-4-5.
Each agent = one real `claude -p --output-format json` call with that agent's
framed prompt + the repo's code as context. No files edited — this measures the
token cost of the *routing decision*. API spend for the run: **$1.43**.

Crashed on a transient null-stdout from the CLI at call 16 → **3 tasks complete,
1 partial**. Not re-run (superseded by other work).

## Result (3 complete tasks)

| task | agents: fixed → SNN | output tok: fixed → SNN | cost: fixed → SNN | tok saved | cost saved |
|---|---|---|---|--:|--:|
| typo fix | 7 → 3 | 34,902 → 17,071 | $0.469 → $0.211 | 51% | 55% |
| add `--port` flag | 7 → 3 | 13,789 → 7,152 | $0.363 → $0.162 | 48% | 55% |
| add unit tests | 7 → 4 | 19,105 → 16,155 | $0.390 → $0.249 | 15% | 36% |
| **total (3)** | | **67,796 → 40,378** | **$1.221 → $0.622** | **40%** | **49%** |
| "make the alerting better" (ambiguous, partial) | 7 → **1** (Clarifier only, stop) | — | — | ~90% | ~90% |

## Read

- **Yes, it saves tokens** — ~40% fewer output tokens / ~49% lower cost on concrete
  tasks vs running the full 7-agent roster, and ~90% on an ambiguous task (one
  Clarifier call instead of a whole mis-aimed pipeline).
- The saving tracks how many agents `score_task()` gates off: 15% when 4/7 fire,
  ~50% when 3/7 fire, ~90% when 1/7 fires.
- **A plain conditional router would achieve the same.** The saving comes from
  conditional execution, not from the SNN being a token-efficiency mechanism.
  Its value is that routing/veto/spawn/promotion all live in one composable,
  learnable topology.

## Caveats

- Input tokens were cache-dominated (shared repo context served from the CLI's
  prompt cache) so the `input_tokens` field read ~9 on every call; the real
  per-call input cost is folded into `total_cost_usd` (cache-read is cheap but
  scales with call count, so cost still tracks agent count).
- haiku with loose "implement/review this" framing produced large, noisy outputs
  (Implementer emitted 10,955 tokens on a *typo* fix). A real constrained
  pipeline would be tighter — but the ratio is what matters, and skipped agents
  emit zero regardless.
- This does not measure code-change quality, nor the risk that `score_task()`
  under-routes a task that needed more review.
