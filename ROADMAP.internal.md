# Internal Roadmap

> **INTERNAL — DO NOT PUBLISH.** This file is for private design notes only.
> Drop it (or `git rm` it) before merging this branch to `main`.

---

## #3 — Ambient mode (experimental, not yet built)

**One-liner:** opt-in background intelligence that narrates cell outputs only when something looks off, instead of requiring the user to call `explain()`.

### Goal
Shift sabre from an on-demand library call to a passive "second pair of eyes" that speaks up only when worth it. The failure mode of every ambient AI tool is being chatty without being useful — the whole design hinges on staying silent 90% of the time.

### Proposed architecture (two-layer)

1. **Triage pass (cheap, per-cell).** Every cell's output goes through a Haiku call that returns strict JSON: `{"flag": "leakage" | "anomaly" | "quality" | null, "one_liner": "..."}`. Default output is `null` → silent.
2. **Expand on demand.** Flagged cells render a small chip (`⚠ leakage suspected`) with an "Explain" button. Click → full `explain()` on the same object. User controls when to spend Opus-level tokens.

Composition note: triage inherits the active domain pack (#6). `use("ml-training") + enable_ambient()` becomes a passive leakage detector. This is the killer combo — don't ship ambient without it.

### Implementation sketch

- `src/sabre/ambient.py`
  - `enable_ambient(budget_usd=1.00, triage_model="claude-haiku-4-5", full_model=DEFAULT_MODEL)`
  - `disable_ambient()`
  - Registers a `post_run_cell` hook (similar pattern to `session.py`)
- `src/sabre/pricing.py` (new) — per-model `{input_per_mtok, output_per_mtok}` table
- `SpendLedger` singleton: records tokens from each response → running cost → flips `enabled=False` at cap, emits toast
- Reuse `agent.py`'s widget rendering for the chip + button

### Open design decisions

| Question | Current lean | Why |
|---|---|---|
| Async vs blocking triage | **Blocking** | Haiku is ~300ms; async adds output-ordering confusion |
| Click-to-expand vs auto-expand | **Click** | Auto can blow the budget on noisy notebooks |
| Triage model | **Haiku 4.5** | Cheap + fast + smart enough |
| Pattern-match shortcut (skip LLM) | **No** | Semantic reasoning is sabre's whole pitch |
| Inherit session memory (#2)? | Undecided | Helps triage context but ~2× cost per call |

### Key risks

- **Chattiness.** Most cells aren't interesting. If triage flags >10% of cells, we've failed.
- **Latency.** Even ~300ms blocking adds up across a long notebook.
- **Privacy.** Every cell output sent to Claude by default. Needs a visible indicator + easy kill switch.
- **Correctness.** A triage false negative hides a real issue the user trusted sabre to catch. Worse than false positives.
- **Cost surprise.** Budget ledger must be bulletproof and visible at all times.

### Prerequisites

- [x] #6 domain packs (shipped on this branch, `966bdc5`) — ambient inherits active pack
- [x] Widget rendering pattern (#1 in `agent.py`) — reused for the chip
- [ ] Pricing table + spend ledger
- [ ] `post_run_cell` hook that makes API calls (session.py's is in-process only — need to vet the threading / error-swallow semantics again)

### Out of scope (at least v1)

- Auto-execute of suggested fixes
- Cross-notebook memory
- Non-LLM pattern-match-only triage path
- User-customizable flag taxonomy (keep fixed at `leakage | anomaly | quality | null`)

### Ship criteria

- Silent default (< 10% flag rate on a representative notebook)
- P95 triage latency < 500ms
- Budget cap verified to auto-disable under load
- `disable_ambient()` is instant (unregisters hook, kills any in-flight calls)
- Clearly labeled **experimental** in README — this is the feature most likely to drive "too chatty" or "too expensive" complaints

---

## Other ideas (parking lot, not prioritized)

- **Undo / correct Claude's narration.** Users correct a misreading; sabre learns within the session.
- **Multimodal inputs beyond matplotlib.** Plotly, Altair, seaborn, PIL, geopandas, audio.
- **Tool-use loop.** Real agent: Claude proposes, runs in a sandboxed kernel, sees the result, iterates.
- **Team/shared context.** Upload the narrated notebook to a shared surface for async review.
- **Eval harness.** Fixtures of (input, expected flags) per domain pack to regression-test prompt changes.
