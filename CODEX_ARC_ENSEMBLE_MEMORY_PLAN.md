# ARC-AGI-3 Flash + Hybrid Ensemble Experiment

## Objective

Design and test a conservative portfolio controller that keeps the protected Flash-Next/MTP c28 agent as the primary policy and uses the Hybrid REPL agent only as a bounded recovery policy when Flash shows verified failure symptoms.

Current public evidence:

- Flash-Next/MTP c28: public score **3.40**; protected incumbent, submission `56170646`.
- Hybrid REPL agent: public score **3.09**; useful as a possible fallback, not a replacement.
- Zero-change repeat-guard run: strict 25-game offline audit passed with offline mean **6.9547**, versus baseline offline mean **5.5027**. This is diagnostic evidence only, not a leaderboard score.

Do not submit or overwrite the incumbent automatically.

## Core hypothesis

Flash is stronger overall, but some failures may be recognizable during execution: repeated no-progress actions, invalid/unavailable actions, stale terminal state, tool-loop exhaustion, or prolonged zero-token continuation. The Hybrid agent may recover some of those states because it uses a different interaction strategy.

The experiment should test:

> Flash primary + one bounded Hybrid recovery handoff, triggered only by verified runtime failure signals, can improve lower-tail game performance without disturbing healthy Flash trajectories.

This is not a voting ensemble and not an average of actions.

## Required architecture

```text
Game Controller
├── Flash-Next/MTP primary agent
├── structured per-game memory
├── progress/no-progress monitor
├── deterministic handoff selector
└── Hybrid REPL recovery agent
```

The controller owns:

- game and level boundaries;
- action validity checks;
- progress tracking;
- memory reset and compaction;
- time and action budgets;
- handoff count;
- terminal handling;
- audit logging.

Neither agent may independently decide to switch the other agent on.

## Handoff policy

Start every game with Flash.

Permit at most **one Hybrid handoff per game**. Do not alternate agents repeatedly.

Trigger a handoff only when all required evidence for one failure class is present:

1. **Repeated no-progress:** a singleton action was executed, produced confirmed `board_changed == False`, zero reward, no level completion, no terminal state, and the same failure pattern persists across a bounded window.
2. **Invalid-action failure:** repeated unavailable or rejected actions, after normal action validation has been applied.
3. **Stale terminal/tool state:** the environment reports a terminal/reset transition but the current agent continues producing terminal-incompatible actions.
4. **Tool-loop exhaustion:** the analyzer repeatedly produces no executable environment action or reaches its bounded tool-turn limit.
5. **Timeout pressure:** the current agent is consuming its action/time budget without progress and the remaining recovery budget is sufficient.

Do not trigger merely because:

- an action repeats but changes the board;
- a valid action has zero generated tokens in isolation;
- the model writes a long response;
- the current score is temporarily low;
- a game has a mechanic that intentionally requires repeated actions.

If progress metadata is missing or ambiguous, do not infer no-progress. Record the event and preserve baseline behavior.

## Recovery contract

When handing control to Hybrid, create a fresh Hybrid context. Do not copy the raw Flash transcript.

Pass only a compact normalized state:

```json
{
  "game_id": "opaque-runtime-id",
  "level": 2,
  "levels_completed": 1,
  "score": 0.42,
  "state": "NOT_FINISHED",
  "current_frame": "serialized-current-observation",
  "valid_actions": ["..."],
  "last_action": "MOUSE(row=..., col=...)",
  "last_action_result": {
    "executed": true,
    "board_changed": false,
    "reward": 0.0,
    "done": false,
    "level_completed": false,
    "game_over": false
  },
  "no_progress_streak": 4,
  "handoff_reason": "repeat_no_change_guard"
}
```

The Hybrid agent must re-ground from `current_frame` before acting. Do not synthesize an action for it and do not rewrite its chosen action except for normal environment validity checks.

Recovery budget:

- configurable maximum of 10–20 actions or an equivalent short time budget;
- no more than one recovery phase per game;
- preserve the original global deadline and teardown contract;
- if Hybrid reaches terminal state, keep its result;
- if Hybrid fails or exhausts its recovery budget, stop or return to Flash only if the runtime has a tested safe re-grounding path.

Default first experiment: **one-way Flash → Hybrid handoff**, no return to Flash. This makes attribution and state ownership simpler.

## Persistent memory policy

### Within one game

Persist compact structured state across analyzer turns:

- latest observation/frame;
- frame fingerprint if available;
- level and score/progress;
- valid actions;
- recent action/result summaries;
- no-progress and invalid-action counters;
- discovered mechanic facts;
- current hypothesis about the game.

Do not persist unlimited raw transcripts. Bound the memory size.

### Between levels in one game

Keep general mechanic facts, but invalidate frame-specific facts:

Keep:

- “red objects respond to mouse clicks”;
- “the switch must be toggled before the target appears.”

Discard:

- exact coordinates from the previous level;
- old frame hashes;
- stale valid-action assumptions;
- terminal-state assumptions.

Require a fresh observation after every level transition or reset.

### Between games

Reset all game-specific memory. Do not carry raw context between games.

Colors, actions, and mechanics may have different meanings in different games. Cross-game memory would also harm reproducibility and could overfit the public game set.

### Between submissions

Do not persist runtime memory from one Kaggle run into another. If a general mechanic library is ever introduced, it must be a separately versioned, immutable artifact and must be evaluated as a new candidate.

## Instrumentation

Record in offline diagnostics and local replay artifacts:

- agent owner: `flash` or `hybrid`;
- handoff reason;
- handoff count;
- action count before and after handoff;
- no-progress streak;
- invalid/rejected action count;
- zero-token streak as a diagnostic only;
- level and score before/after handoff;
- terminal outcome;
- wall-clock and remaining budget;
- whether memory was reset or compacted.

Submission mode must retain the existing privacy boundary. Emit only aggregate health counters, not game IDs, frames, scores, prompts, or transcripts.

Suggested aggregate submission-health fields:

```json
{
  "flash_games": 25,
  "hybrid_handoffs": 0,
  "hybrid_terminal_completions": 0,
  "guard_trigger_count": 0,
  "invalid_action_count": 0,
  "max_no_progress_streak": 0,
  "timeouts": 0,
  "teardown_ok": true
}
```

## Implementation constraints

- Start from a clean copy of the active Flash source path.
- Preserve the protected 3.40 package and all unrelated working-tree changes.
- Do not change model weights, prompts, sampling, serving profile, concurrency, scorer, or teardown in the first ensemble experiment.
- Keep Flash and Hybrid implementations isolated behind adapters.
- Use exact source anchors and digest-bound overlays where external/pulled source is patched.
- Fail closed if the intended dispatch boundary is missing or duplicated.
- Do not add cross-game memory.
- Do not submit to Kaggle as part of local implementation.

## Evaluation plan

### Stage A: replay and unit tests

Build deterministic fake environments covering:

- valid repeated action that changes the board;
- confirmed repeated no-change action;
- changed valid-action set;
- invalid/unavailable action;
- terminal result;
- level completion;
- missing progress metadata;
- one-way Flash → Hybrid handoff;
- handoff reset between games;
- bounded recovery exhaustion.

### Stage B: paired offline diagnostics

Compare against a contemporary c28 Flash control using identical game order and runtime settings.

Report:

- per-game score delta;
- wins/losses/ties;
- median and lower-tail score;
- action count;
- timeout/cancellation count;
- invalid-action count;
- guard/handoff count;
- score before and after handoff;
- protected-game regressions.

Do not select the candidate from offline mean alone.

### Stage C: strict package gates

Require:

1. generated notebook compilation;
2. clean metadata and isolated experiment identity;
3. one-game or bounded preflight audit;
4. complete 25-game full-offline audit;
5. clean teardown and zero unexpected cancellations;
6. manual review of every handoff-triggered game.

### Stage D: competition decision

Only submit an exact completed/audited kernel version if the experiment shows a plausible transfer mechanism and no unacceptable regression. Promotion requires a public score strictly above **3.40**. Keep the incumbent otherwise.

## Acceptance criteria for the first ensemble candidate

Accept for a public probe only if all are true:

- no Flash behavior changes before a verified handoff;
- at most one handoff per game;
- no cross-game memory leakage;
- no increase in invalid actions or cancellations;
- no teardown/watchdog regression;
- no regression on protected known-good games in paired diagnostics;
- handoffs are rare, explainable, and manually inspectable;
- complete strict 25-game audit passes.

A higher offline mean alone is not sufficient.

## Codex implementation deliverables

Codex should return:

1. changed files;
2. architecture summary;
3. exact handoff trigger conditions;
4. memory reset/compaction rules;
5. tests added and their results;
6. generated package path;
7. risks and known limitations;
8. confirmation that no Kaggle upload or competition submission occurred.
