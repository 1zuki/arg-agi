# ARC-AGI-3 low-resource plan

## Decision

Keep the combined notebook as a scored-run artifact and use CPU checks only for
its notebook, patch, packaging, and audit contracts. Do not treat a CPU pass as
evidence that the model can play ARC-AGI-3 or that a leaderboard score will
improve.

Start with generated games and scripted-agent evaluation when compute is
available locally. Defer fine-tuning until there is a GPU budget, a held-out
transfer test, and a specific failure mode that training is expected to fix.

## What is testable now

- notebook JSON and ordinary Python-cell compilation;
- patch and builder tests that do not import the large model;
- output schema, provenance, teardown, and audit logic;
- small deterministic game or harness probes that run without the Qwen server.

These checks are useful for catching regressions in the artifact. They cannot
measure VRAM fit, SGLang/MTP startup, token throughput, interactive gameplay,
or a public score.

The repository also has a side-effect-free hardware gate:
`python scripts/arc3_gpu_preflight.py`. It only queries `nvidia-smi` and
reports whether the host meets the planning thresholds; it does not load model
weights or start a server. A pass is necessary but not sufficient for the
exact one-game serving preflight.

## Why the GPU gate is real

The active profile expects the attached quantized Flash-Next target and draft
models, CUDA/SGLang, and an RTX PRO 6000 Kaggle runtime. The local RTX 4050 has
about 6 GiB of VRAM, while the target model assets are tens of gigabytes, so
CPU or local-GPU execution is not a representative substitute. RAM and swap do
not make that serving configuration valid.

Even on the RTX PRO 6000, this is a long run: the notebook declares a 32,400
second (nine hour) wall-clock budget. Its offline branch schedules 25 public
games concurrently with a 25-minute per-game limit; its competition branch
uses the gateway's available game list, concurrency 120, and a 532-minute
per-game limit. Setup, model loading, gateway startup, game play, output
finalization, and teardown all compete for the wall-clock budget. A run can
therefore start correctly yet fail to finish, time out, or receive a weak
public score.

Historical Kaggle submissions show the combined-profile family has been run
successfully, but its public scores varied substantially: Versions 1, 2, 3, 4,
6, 7, and 8 scored 29.49, 22.62, 29.69, 22.80, 29.43, 0.00, and 0.00
respectively. These are separate completed submissions, not a controlled
repeatability study. Keep the combined-profile scores separate from the
Flash-Next/MTP c28 family, whose protected result is 3.40. Compare a new
combined-profile run to prior combined-profile artifacts and only use a
completed public result to make score claims. Offline means and partial traces
are diagnostic only.

## Smallest useful GPU sequence

When equivalent GPU access returns, use one hypothesis and one frozen package:

1. Run a one-game preflight with the exact model inputs and serving settings.
   Record startup, health, VRAM, action count, teardown, and output hashes.
2. Run all 25 offline games and require a complete, audited schedule under the
   offline branch's actual concurrency and deadline. Reject partial or
   cancelled runs even if their offline mean looks high.
3. Run the competition rerun only after the first two gates pass. Promote only
   if the completed public result improves on the best completed result for the
   same candidate family under comparable settings.

This sequence separates infrastructure failure, gameplay behavior, and public
evaluation. It also avoids spending a nine-hour slot on an unbounded change.

## Game generation before fine-tuning

The public Skyfall ARC-AGI-3 game generator provides an end-to-end idea,
specification, build, blind-audit, structural/visual/seed-gate, and play-test
workflow: <https://github.com/Skyfall-Research/arc-agi-3-games>. NVIDIA's
DreamTeam repository also contains an ARC-AGI-3 `game_creator`:
<https://github.com/NVIDIA/dream-team>.

The Tufa Labs public artifact is the Duck/TAAF solver harness. The available
evidence does not show Tufa publishing the standalone Skyfall generator; the
Skyfall project cites Tufa's ARC-3 prompts as reference material. ARC-AGI-2 is
related, but its usual static grid input/output contract is different from
ARC-AGI-3's interactive game and action loop.

Use generated games for cheap tests of action schemas, state tracking, reset
and recovery behavior, and harness instrumentation. Split by mechanic family
and random seed so the evaluation measures transfer to unseen games. Do not
convert success on generated games into an official ARC-AGI-3 score claim.

The repository now contains a small CPU-only probe at
`scripts/arc3_generated_game_probe.py`. It generates three deliberately simple
mechanic families (`direct`, `key`, and `switch`) and checks the generator-known
path across train-like and held-out seeds. The probe currently covers 15 games,
45 levels, and 45/45 oracle level completions in a few milliseconds. The
companion tests cover byte-stable same-seed generation, changed layouts for
different seeds, reset restoration, invalid-action rejection, held-out seed
reporting, and mutable-state isolation. These are harness and reproducibility
checks only; the oracle has generator access and therefore says nothing about
an agent discovering the rules.

The same module now includes a blind scripted explorer. It receives only a
wall/agent board, level number, and action feedback; it does not receive the
goal, family, special cell, or unlock flag. With a 512-action cap it clears the
held-out seeds used by the local probe across all three families. This is still
an intentionally small search benchmark, but it is a more meaningful
regression target than the generator oracle because hidden-state leakage is
tested explicitly.

Three explicit observation-only candidates are now compared under the same
budget: `TwoPhaseExplorer` defers toggle probes until movement is exhausted,
`PublicCorridorExplorer` follows a far-side public-grid prior, and
`AdaptiveCorridorExplorer` falls back to two-phase exploration when that prior
does not complete the level. The adaptive candidate is the strongest generated
diagnostic, but none is promoted into the scored notebook without an official
transfer run.

The probe also includes an observable sequence family. It exposes progress
feedback while hiding the ordered interaction cells and total, so
`SequenceExplorer` must discover a multi-step objective from public state. On
eight geometries and three held-out seed splits it completed 2,400/2,400 games
at a 1,024-action cap; this remains a toy diagnostic about observation design,
not evidence of ARC-AGI-3 transfer or a reason to alter the scored notebook.

The delayed-feedback extension keeps the ordered interaction state hidden and
delays the public progress signal by two or three actions, with deterministic
one-step noise every fifth step. At a matched 1,536-action budget,
`SequenceExplorer` completed 2,400/2,400 games and 7,200/7,200 levels across
the same eight geometries and three seed splits; blind, adaptive, two-phase,
and corridor completed 1,449, 1,022, 14, and 1 games respectively. This is a
stress test for stale feedback in the CPU harness. It is not an official score
or a justification for changing the scored notebook. Records are in
`build/arc3-policy-benchmark-20261031/delayed_feedback/summary.json`.

It also includes a small recovery family with hidden hazards. Hazards are not
shown in the public board; the policy only sees the life-loss/reset feedback
and must remember the failed transition. This exercises a failure mode that
replay-only trajectories cannot cover. The current probe wins its two held-out
recovery games within a 1,024-action cap, with recorded life losses. The
adaptive and two-phase variants reduce recovery actions on the canonical family
while preserving 100/100 wins on the archived split. That is a harness
regression result, not evidence of transfer to official games.

Run the bounded probe with:

```bash
python scripts/arc3_generated_game_probe.py --seeds 0 1 2 3 4
python scripts/arc3_generated_game_probe.py --blind-policy --seeds 101 102
python scripts/arc3_generated_game_probe.py --two-phase-policy --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry center_goal --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry random_goal --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --delayed-feedback-benchmark --geometry random_goal --seeds 101 102 --levels 3 --max-actions 1536
python scripts/arc3_path_hazard_matrix.py --output-dir build/arc3-policy-benchmark-20261101/path_hazard
python scripts/arc3_generated_game_probe.py --budget-curve --seeds 2000 2001 2002
python -m unittest tests.test_arc3_generated_game_probe -v
```

The budget curve is the useful metric for this family. On a 100-seed,
five-level held-out sweep, the blind policy reached 90.3% at 256 actions,
95.7% at 512, and 95.7% at 1,024; the direct family was 100% at 512 while
key/switch remained 92%/95%. This is evidence of an exploration-cost tradeoff
inside the toy family, not evidence of transfer to official games.

The non-far `near_goal`, `offset_goal`, and `random_goal` layouts now test
arbitrary visible start/goal placement. Across three held-out seed splits,
two-phase and adaptive each reached 300/300 games at 512 actions for all three
layouts, while the corridor prior regressed sharply on random goals. A
model-backed result should be considered useful only if it improves held-out
performance under a fixed action and time budget; these generated results must
not be used as an official score or a training shortcut.

The same three layouts were replayed with hidden hazards and reset feedback at
a 1,024-action cap. Two-phase and adaptive retained 100/100 wins on each
layout; the corridor prior failed most often on random goals. This is a
recovery-contract result, not evidence that the policy transfers to private
games.

The `path_hazard` extension makes that recovery test stricter: one hidden hazard
is placed on the nominal route, while the generator guarantees an alternate
public route remains available. The reusable matrix runner covers eight
geometries, three held-out seed splits, 2,400 games, and 7,200 levels at a
1,024-action cap. Adaptive and two-phase each completed 2,400/2,400 games;
baseline also completed 2,400/2,400 but used 100.889 mean actions versus
adaptive's 50.093 and two-phase's 57.502. Corridor completed 1,879/2,400 and
exhausted its budget on 521 games. This isolates hidden fatal-path recovery in
the CPU harness and remains diagnostic only. The aggregate and per-split
records are in `build/arc3-policy-benchmark-20261101/path_hazard/`.

The path-hazard runner also supports a two-hazard stress case. With two hidden
route hazards and an alternate public route preserved, the same 2,400-game /
7,200-level matrix kept adaptive and two-phase at 100% wins, while corridor
fell to 1,857/2,400 with 543 budget exhaustions. This tests repeated recovery
only inside the CPU harness. A paired immediate-fallback ablation reduced mean
actions for two hazards but increased life losses for both one- and two-hazard
cases, so it was not promoted.

### Read-only Skyfall shelf evidence

The local Skyfall shelf clone can be audited without invoking its generator or
spending on an agent API:

```bash
/tmp/arc3-cpu-smoke/bin/python scripts/arc3_skyfall_shelf_smoke.py \
  --shelf-root /tmp/arc-agi-3-games-research
```

The audit checks the packaged manifest claims, re-derives every validated
layout twice to check determinism and distinctness, checks action and
observation schema, and replays each canonical trajectory at its authored
layout seed. The current shelf contains 11 games and 85 levels; all 55
validated seed layouts passed the deterministic/distinctness check, and all 11
canonical trajectories won across 2,266 actions. The extra 44 seed wins are
manifest claims because the shelf does not include their solver trajectories.
These results describe package and environment integrity only. They are not a
blind-agent result, an OOD transfer result, or evidence that fine-tuning on
the trajectories will help.

## Fine-tuning gate

Fine-tuning is currently deferred. It would add data-generation, adapter,
training, and evaluation cost without local hardware or evidence that the
synthetic distribution transfers to private ARC-AGI-3 games. If revisited,
require all of the following:

- a concrete failure mode and a proposed training signal;
- mechanic-family and seed-held-out splits;
- base-model versus adapter comparison under the same action/runtime budget;
- a cheap generated-game result that survives the held-out split;
- an exact ARC-AGI-3 preflight before any competition rerun.

Until those gates exist, generator-first evaluation has the better information
per unit of compute.
