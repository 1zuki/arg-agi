# ARC-AGI-3 Optimization Progress

Last verified: **2026-11-03 11:43 (Asia/Saigon)**

This file is the compact continuity record for the combined Milestone 2
optimization. Keep detailed traces and benchmark JSON in `build/` and other
machine-readable artifacts; update this file whenever a score, gate, or
material experiment changes.

## Current decision

- **Protected best validated result:** combined-profile Kaggle submission
  `57385516`, kernel version **3**, public score **29.69**.
- **Promotion rule:** a candidate must be a completed, schema-valid, audited
  run and beat 29.69 on the public leaderboard. Offline means and CPU probes
  do not satisfy this rule.
- **Current local candidate:**
  `arc-agi-3-milestone-2-solution-combined.ipynb` with the tracked hardening
  changes in this repository. The unvalidated `ARC3_SUMMARY_*` runtime toggles are now disabled;
  summary support remains in the harness patch as an opt-in diagnostic. The
  protected 12-minute server-start timeout is restored. The bundle's one-line
  legacy seed normalization, output contract, and offline scaffold checks remain
  enabled.
- **CPU runtime compatibility gate:** an isolated Python 3.12 environment using
  the local `arc-agi==0.9.8` wheelhouse reset and accepted one action on all
  25 bundled games (`25/25` resets and `25/25` actions, zero errors). This is
  SDK/environment evidence only; it does not validate model gameplay or score.
- **Harness patch provenance gate:** the embedded cell-2 patch applies cleanly
  to the exact pinned TAAF baseline `7652836056c59e044f093e3c13ed7438c814169e`
  after removing one legacy `seed=` argument that cell 4 normalizes at runtime.
  The dirty external checkout was not used as a baseline.
- **Repeatability gate:** `scripts/check_combined_harness_patch.py` now performs
  that provenance check in a temporary archive tree, so the external checkout
  is never modified. Its reproduced report has the same patch hash and
  `git_apply_check=true`.
- **Latest known rerun:** submission `57747556` (combined profile v13) is
  `COMPLETE` at **8.98**. It is rejected for promotion and does not change the
  29.69 incumbent.
- **Latest remote run, kernel version 1:**
  `izukia/arc-agi3-m2-combined-profile-full` reached `KernelWorkerStatus.COMPLETE`.
  Kaggle's pulled notebook had the same 23 cell sources as the uploaded package.
  The browser-verified output preview has been recovered under
  `build/combined-kaggle-full/audit-version-1-20261102/`: it is explicitly
  offline-only (`true_submission=false`) with a zero-row Parquet scaffold, 25
  terminal games, **0 wins**, mean score **4.396**, and median **0**. The
  recovery manifest records that the original binary/log export is unavailable
  locally after package replacement; this directory is evidence-preserving, not
  a byte-for-byte output archive.
- **Active combined run contract:** the recovered `combined_run_contract.json`
  confirms 25 unique, terminal offline game runs and a valid four-column Parquet
  scaffold with zero rows and `true_submission=false`, as expected for offline mode. The
  mean score was **4.396**, median **0**, and all runs ended `gave_up` under the
  25-minute per-game offline cap. The earlier browser audit observed read
  timeouts shrinking with the remaining per-game budget; this short offline
  result is not comparable to the 532-minute competition-mode per-game cap. The generic
  `scripts/audit_kaggle_output.py` is documented for the older checkpoint-8
  workflow and expects different manifest/log filenames, so its failures do
  not override the active notebook's contract. The submission-mode audit of
  this offline artifact is not a valid competition-mode audit.
- **Competition-mode code submission:** submission **`57910947`** for
  `izukia/arc-agi3-m2-combined-profile-full`, kernel version **1**, reached a
  terminal successful run with public score **6.92**. It is rejected for
  promotion because it is below the protected **29.69** incumbent. The saved
  Kaggle code output is still the standard offline run (`true_submission=false`,
  25 terminal games, zero-row scaffold); Kaggle does not expose a separate
  competition-mode artifact on that page, so the hidden rerun cannot be audited
  beyond its terminal public score.
- **Final-selection state:** the competition page is past its 2026-11-03
  06:59 (Asia/Saigon) deadline. It shows **0/2** manually selected and disabled
  selection controls, so Kaggle will auto-select up to two public best-scoring
  unselected submissions. Keep `57385516` at 29.69 protected; the next public
  candidate in the visible history is the 29.49 combined version-1 result.
- The CPU-only observation-policy candidates, observable and delayed-feedback
  mechanics, and non-submission controls remain diagnostic evidence; they are
  not leaderboard claims and are not promoted into the scored notebook.
- Objective requirement audit:
  `build/arc3-optimization-audit-20261101/requirements-audit.json` records the
  completed offline run, the terminal 6.92 public score, the failed promotion
  gate, and the locked final-selection state. The offline GPU result is not
  treated as competition transfer evidence.

## Best validated configuration

The version-3 result belongs to the combined Milestone 2 family. The current
notebook records the winner profile below; verify byte identity and package
hashes before claiming that a future run reproduces version 3.

### Notebook and serving profile

- Notebook: `arc-agi-3-milestone-2-solution-combined.ipynb`.
- Default: `ARC3_CROSS_SOLUTION_PROFILE=winner` (`dfranzen-27.89-winner`).
- Model: Qwen3.8-Flash-Next W4A16 AutoRound; MTP draft checkpoint from the
  pinned Albucino input.
- Serving: Pennyroyal/SGLang v2.5.3 on the RTX PRO 6000 Kaggle runtime;
  TAAF/Duck harness patch based on Tufa Labs commit `7652836`.
- `server_context=139264`; `harness_context=131072`;
  `max_output_tokens=12288`; `context_drain_tokens=59392`.
- `history_assistant_turns=150`; `history_turn_drain=30`;
  `history_drain_coalesce=true`.
- `max_active_streams=10`; `max_running_requests=10`; `mamba_cache=60`.
- `mem_fraction_static=0.96`; `chunked_prefill_size=8192`;
  `kv_cache_dtype=fp8_e4m3`; `temperature=0.7`; `image_upscale=10`.
- Speculative decoding is enabled with `SPEC_STEPS=3`; priority scheduling is
  enabled.
- Offline validation branch: 25 bundled public games, 25-minute per-game cap,
  effective concurrency 25, nine-hour (`32400 s`) notebook budget.
- Competition branch: gateway game list, effective concurrency 120,
  `532*60 s` per-game cap, same nine-hour notebook budget.
- Required Kaggle inputs and fail-closed source/model checks are documented in
  `README.md`; do not submit the offline empty `submission.parquet` scaffold.

### Score evidence

| Combined kernel version | Submission | Public score | Decision |
| ---: | ---: | ---: | --- |
| 1 | `57345516` | 29.49 | Valid historical result |
| 2 | `57321574` | 22.62 | Rejected |
| 3 | `57385516` | **29.69** | **Protected incumbent** |
| 4 | `57414229` | 22.80 | Rejected |
| 6 | `57472977` | 29.43 | Rejected |
| 7 | `57514838` | 0.00 | Rejected |
| 8 | `57587807` | 0.00 | Rejected |
| 13 | `57747556` | 8.98 | Rejected; latest v13 rerun |
| 1 (full) | `57910947` | 6.92 | Rejected; terminal score below incumbent |

The live account listing on 2026-10-30 also showed `57797325` at 0.00 for a
different notebook (“The AGI Boys”); it is not attributed to this candidate.
The listing did not expose a reliable remaining-submission count.

## Experiment ledger

### Combined-profile history

- Versions 1 and 3 are the strongest completed combined runs (29.49 and
  29.69). Version 6 reached 29.43; later versions 7, 8, and 13 regressed.
- Version 6/v13 included offline/schema scaffolding and diagnostic hardening;
  those changes did not establish a score improvement. Keep the public score
  and the local artifact contract as separate evidence.
- Current local edits improve bundle discovery, provenance/contract checks,
  output auditing, and documentation. They are reliability changes until a
  comparable public rerun proves a score effect.

### Flash-Next/MTP family (separate lineage)

- Protected Flash result: submission `56170646`, public score **3.40**.
- Later c8/c12, state-overlay, ACTION7, inspection, and repeat-guard variants
  produced useful offline diagnostics but did not beat 3.40 publicly. Do not
  use those results as evidence for the combined profile.
- The recurring lesson is that a higher offline mean is insufficient when the
  run is incomplete, teardown is invalid, or public transfer is poor.

### CPU and package experiments

- Generated-game probe: 15/15 oracle games, 6/6 blind-policy games, and 2/2
  hidden-hazard recovery games on the fixed local probes.
- On the documented 100-seed blind sweep, the toy-family budget curve was
  90.3% at 256 actions and 95.7% at 512 and 1,024 actions. This is a harness
  regression result, not an ARC-AGI-3 leaderboard forecast.
- Read-only Skyfall shelf audit evidence: 11 games, 85 levels, 55 validated
  seed layouts deterministic/distinct, and 11 canonical trajectories replayed.
  Extra seed wins are manifest claims without solver traces.
- Two-phase CPU candidate (`TwoPhaseExplorer`) uses movement-first exploration,
  deferred probes, and a verification pass. At a 512-action budget it reached
  **300/300** games on each held-out split 100–199, 200–299, and 1000–1099;
  the frozen `BlindExplorer` reached 290/300, 287/300, and 290/300. Mean
  actions on those splits were 143.207/141.760/143.593 for the candidate and
  148.413/151.257/149.170 for the baseline. On hidden-hazard recovery seeds
  100–199 at 1,024 actions, both policies won 100/100; mean actions were
  79.53 for two-phase versus 138.94 for baseline. At 128–256 actions the
  candidate has a switch-family tradeoff, so it remains an explicit diagnostic
  option rather than the default policy. Full records and budget curves are in
  `build/arc3-policy-benchmark-20261031/`.
- Public-corridor CPU candidate (`PublicCorridorExplorer`) uses only the public
  grid and observed start position. It selects the geometrically distant
  visible target, probes the union of shortest public-grid corridors, and
  returns to the target. At 512 actions it reached **300/300** games on
  held-out splits 100–199, 200–299, and 1000–1099, with mean actions
  100.603/100.257/99.530 and maxima 201/214/202. Hidden-hazard recovery on
  seeds 100–199 reached 100/100 at 1,024 actions with 79 life losses and mean
  30.47 actions. Its 5-level budget curve reached 299/300 at 320 actions and
  300/300 at 384; at 128–256 actions it trails the frozen baseline and the
  two-phase candidate, so it remains an explicit generated-game diagnostic.
  The geometric target prior is specific to this toy generator and has no
  ARC-AGI-3 transfer evidence.
- Alternate-geometry audit now covers vertical flip, horizontal flip, and 180°
  rotation on the same 100–199 held-out seeds. `PublicCorridorExplorer` and
  `TwoPhaseExplorer` each stayed at **300/300** games at 512 actions for all
  three transforms. The frozen `BlindExplorer` fell to 251/300, 270/300, and
  224/300, showing that its original probing order is orientation-sensitive.
  Corridor means were 102.650/101.633/104.020 actions; recovery stayed 100/100
  for every policy and transform. This supports a public-grid robustness
  improvement while still leaving arbitrary goal geometry untested. Raw records
  and the geometry summary are in
  `build/arc3-policy-benchmark-20261031/geometry/`.
- The adaptive candidate (`AdaptiveCorridorExplorer`) keeps the corridor fast
  path and switches to `TwoPhaseExplorer` when the public geometric target is
  reached without completion. On the new `center_goal` layout, where the board
  extends beyond a hidden center goal, corridor alone fell to 115/300 while
  adaptive recovered **300/300** at 512 actions with mean 102.883. On the
  canonical and three transformed layouts adaptive matched corridor at
  300/300. Its 5-level center-goal budget curve reached 299/300 at 320 and
  300/300 at 384; this is the strongest synthetic policy so far, but still a
  generated-game diagnostic with no ARC-AGI-3 transfer evidence.
- The arbitrary-layout extension adds `near_goal`, `offset_goal`, and
  `random_goal` start/goal layouts. Across held-out splits 100–199, 200–299,
  and 1000–1099 at 512 actions, both `TwoPhaseExplorer` and
  `AdaptiveCorridorExplorer` reached **300/300** games for every layout family.
  On split 100–199, the frozen baseline reached 202/300, 254/300, and 236/300;
  the corridor prior reached 125/300, 151/300, and 38/300 respectively. Mean
  actions for two-phase/adaptive were 87.490/86.413 (`near_goal`),
  120.340/101.693 (`offset_goal`), and 100.190/112.203 (`random_goal`). This
  is stronger generated-game coverage and a measured failure of the corridor
  prior outside far-side goals, not evidence for hidden-game transfer or a
  promotion into the scored notebook. Records are in
  `build/arc3-policy-benchmark-20261031/geometry/non_far/summary.json`.
  The five-level budget curve on seeds 100–199 shows adaptive reaching
  300/300 by 384 actions for `near_goal` and `offset_goal`, and by 512 for
  `random_goal`; two-phase reaches 300/300 at 512 for all three. The corridor
  prior plateaus at 89/300, 113/300, and 14/300 across those layouts. Curves
  are in `geometry/non_far/budget_curves_100_199.json`.
- Hidden-hazard recovery on the same 100–199 seeds at 1,024 actions kept
  two-phase and adaptive at **100/100** for `near_goal`, `offset_goal`, and
  `random_goal`. Corridor recovery fell to 75/100, 80/100, and 24/100 while
  consuming 293.15/235.98/784.52 mean actions; two-phase means were
  17.74/54.57/37.92 and adaptive means 59.08/47.86/76.48. This confirms the
  added layouts exercise the recovery path as well as clean-board completion.
  The machine-readable recovery report is
  `geometry/non_far/recovery_summary_100_199.json`.
- The observable sequence-mechanic probe adds an ordered interaction objective:
  progress is public feedback, while the ordered interaction cells and total
  remain hidden. Across eight geometries (canonical, three transforms,
  center-goal, near-goal, offset-goal, and random-goal), three held-out seed
  splits, 100 seeds per split, and a 1,024-action budget, `SequenceExplorer`
  completed **2,400/2,400 games and 7,200/7,200 levels** with no budget
  exhaustion and a mean of 409.701 actions. The baseline completed 1,463/2,400
  games, adaptive 1,061/2,400, two-phase 21/2,400, and corridor 0/2,400.
  This isolates the value of observable multi-step feedback in the toy
  harness; it is a diagnostic only and does not establish ARC-AGI-3 transfer.
  The benchmark materializes iterable seed inputs once so every policy receives
  the same held-out games. Raw per-split records and the aggregate are in
  `build/arc3-policy-benchmark-20261031/sequence/summary.json`.
- The delayed-feedback extension keeps the ordered interaction objective
  hidden while delaying public progress by two or three actions and applying
  deterministic one-step noise every fifth step. At a matched 1,536-action
  budget across eight geometries and three held-out splits (2,400 games,
  7,200 levels), `SequenceExplorer` completed **2,400/2,400 games and
  7,200/7,200 levels** with no budget exhaustion. The baseline completed
  1,449/2,400 games, adaptive 1,022, two-phase 14, and corridor 1. This is a
  stale-feedback stress test in the CPU harness; it does not justify a scored
  notebook change or an ARC-AGI-3 transfer claim. Raw per-split records and the
  aggregate are in
  `build/arc3-policy-benchmark-20261031/delayed_feedback/summary.json`.
- The `path_hazard` extension places one hidden hazard on the nominal route and
  guarantees an alternate public route. The reusable matrix covers eight
  geometries, three held-out seed splits, 2,400 games, and 7,200 levels at a
  1,024-action budget. Adaptive and two-phase each completed **2,400/2,400**
  games with mean actions **50.093** and **57.502**; baseline also completed
  2,400/2,400 but used **100.889** mean actions. Corridor completed 1,879/2,400
  and exhausted its budget on 521 games. This is a stricter synthetic recovery
  diagnostic, not a scored-notebook or ARC-AGI-3 transfer result. The aggregate
  is `build/arc3-policy-benchmark-20261101/path_hazard/summary.json`.
- The two-hazard extension stress-tests repeated hidden failures while keeping
  the public grid and an alternate route valid. Across the same 2,400 games and
  7,200 levels, adaptive and two-phase retained **2,400/2,400** wins; baseline
  also retained 2,400/2,400; corridor fell to **1,853/2,400** with 547 budget
  exhaustions. Adaptive and two-phase used 64.405 and 63.515 mean actions,
  versus baseline 105.515. The route invariant passed for all 7,200 levels.
  This remains CPU-only generated-game evidence. The aggregate is
  `build/arc3-policy-benchmark-20261101/multi_path_hazard/summary.json`.
- A paired adaptive-recovery ablation compared the current adaptive policy with
  immediate fallback after the first life loss over one- and two-hazard cases
  (4,800 games and 14,400 levels total). Immediate fallback was slower on one
  hazard (52.205 vs 50.093 mean actions) and faster on two (60.453 vs 64.405),
  but increased mean life losses in both cases (2.703 vs 1.590; 4.750 vs
  3.101). Win rates tied at 100%, so the variant is rejected for promotion.
  The reproducible ablation is
  `scripts/arc3_adaptive_recovery_ablation.py`; its aggregate is
  `build/arc3-policy-benchmark-20261101/adaptive_ablation/summary.json`.
- The active combined notebook now validates post-run artifacts before treating
  execution as successful. Its inline contract rejects missing or malformed
  `benchmark.json`, nonterminal game states, nonfinite/out-of-range scores,
  action-history mismatches, duplicate game IDs, malformed Parquet framing,
  wrong Parquet columns, missing competition rows, empty IDs, non-boolean
  terminal flags, nonfinite/negative row scores, duplicate row IDs, and
  non-empty offline scaffolds; it writes aggregate `combined_run_contract.json`
  metadata. The contract helpers are covered by the notebook contract tests and
  do not change the model, prompt, or serving profile.
- The active combined notebook now implements the documented non-submission
  controls: exact `TAAF_OFFLINE_GAMES` selection, positive bounded
  `TAAF_OFFLINE_MAX_GAMES`, and positive concurrency, analyzer-timeout,
  action-budget, and per-game-runtime overrides. Unknown game ids, malformed
  values, zero values, and runtimes above the notebook budget fail closed;
  competition settings remain fixed. Focused tests cover selection, bounding,
  and parser rejection.
- The scored full-kernel rerun returned **6.92**, below the protected 29.69
  result. Its saved output is the separate offline diagnostic artifact, so this
  score does not identify whether the decline came from the runtime candidate or
  Kaggle's competition infrastructure. The unvalidated rolling-summary profile
  is therefore disabled in the local final candidate and remains opt-in only.
- The final candidate also restores the protected `SERVER_STARTUP_TIMEOUT = 12
  * 60` setting; the earlier 60-minute wait was an unvalidated local change
  that could consume the fixed notebook budget before gameplay.
- A handoff prototype that copied the corridor's covered/probed cells into the
  two-phase fallback was rejected. It preserved 300/300 on canonical,
  center-goal, near-goal, and offset-goal layouts, but fell to 299/300 on
  `random_goal`; it also worsened mean actions on center and near layouts.
  The active adaptive policy therefore keeps the conservative replan instead
  of silently promoting an unvalidated state-sharing change.
- Fine-tuning remains deferred: there is no held-out transfer evidence and no
  local hardware capable of the target serving profile.

## Reliability and current gates

- Current checks: full `python -m unittest discover -s tests -q` → **204
  passed, 2 skipped**; final harness-patch provenance check →
  `git_apply_check=true`; final package builds and `git diff --check` pass.
  The package builder compiled every runnable notebook cell and skipped only the
  supported patch-magic cell.
- Hardware gate: `python scripts/arc3_gpu_preflight.py` → **not ready**;
  local NVIDIA GeForce RTX 4050 Laptop GPU has 6,141 MiB total and 355 MiB
  free, below the 32,000/24,000 MiB planning thresholds. No model-serving run
  was started.
- Active package gate: `scripts/build_combined_kaggle_package.py` generated and
  re-read both final package modes. The full package is byte-identical to the
  source notebook (`0e4f2051e03974bfd42a6cdc14cd007f7a843b5bb8de352083c3d4017af6bed9`)
  at `build/combined-kaggle-final-20261103/`; the one-game preflight package
  carries the same source hash and generated notebook hash
  `2e1811bcbd3e8860cf37ad6d9c86ed61790740d252cbee40a0051783bb51e246` at
  `build/combined-kaggle-final-preflight-20261103/`.
  Both manifests pin the RTX Pro 6000 shape, internet-disabled metadata, two
  model inputs, and the active source hash. The preflight slice is injected at
  the offline game-list anchor; the gateway competition branch is unchanged.
- CPU/package checks cover notebook syntax, patch/build/audit contracts,
  schema validation, deterministic probes, the isolated Python 3.12 SDK smoke,
  repeatable patch application against the pinned clean baseline, and the
  path-hazard route invariant. They cannot validate
  Qwen serving, VRAM fit, interactive gameplay, or leaderboard transfer.
- The working tree already contains unrelated dirty and untracked project
  work. Preserve it; this cycle adds the active package builder and tests,
  README/progress documentation, and the machine-readable package artifacts.

## Findings, bottlenecks, and unresolved issues

1. **Equivalent GPU access is the hard gate.** A local RTX 4050 is not a valid
   substitute for the RTX PRO 6000-class serving contract.
2. **Public variance is large.** Combined scores range from 0.00 to 29.69;
   versions are not a controlled repeatability study.
3. **The current local candidate scored below the incumbent.** The 6.92 result
   is terminal public evidence, but its hidden rerun artifact is not exposed;
   the offline output contract proves packaging, not competition gameplay
   quality or the cause of the decline.
4. **The nine-hour budget is tight.** Lower concurrency can create serial
   batches and incomplete runs; every candidate must satisfy the schedule and
   teardown gates before submission.
5. **The final-selection window is closed.** Kaggle shows 0/2 manual selections
   with disabled controls and will auto-select up to two public best-scoring
   unselected submissions.
6. **This environment has no Kaggle CLI or credential file.** Browser evidence
   verified the terminal score and final-selection state; the saved code output
   still does not expose the hidden competition rerun artifact.
7. **Two-phase policy tradeoff is budget-dependent.** It improves completion at
   512+ toy actions and reduces recovery cost, but can delay switch completion
   under small budgets; do not infer a competition policy change from this
   diagnostic alone.
8. **Public-corridor policy is faster at a generous toy budget but prior-bound.**
   It reduces the 512-action mean substantially and preserves recovery, while
   losing to the baseline at 192–256 actions and relying on the generated
   board's far-side target geometry. Keep it as a measured diagnostic until a
   broader public-grid generator or real ARC-AGI-3 transfer test supports it.
9. **Adaptive fallback repairs the tested prior failure.** It preserves the
   corridor result on far-side goals and recovers the center-goal layout, but
   this does not prove recovery for arbitrary hidden mechanics or real games.
10. **Delayed public feedback exposes policy dependence on observation timing.**
    `SequenceExplorer` remains complete across all 2,400 delayed-feedback
    games, while policies that do not track the delayed signal regress sharply.
    The mechanic is synthetic and the unlock state remains immediate, so this
    result is a robustness diagnostic rather than a model or leaderboard gain.
11. **Documented non-submission controls now match the active notebook.** The
    selection and solver overrides are bounded and fail closed, while the
    competition branch ignores them. This removes a reproduction mismatch but
    does not provide gameplay or score evidence.
12. **The local SDK path is usable when isolated from the host environment.**
    With Python 3.12 and the bundled `arc-agi==0.9.8` wheels, every selected
    game reset and one-step action succeeded. This is a compatibility gate, not
    a gameplay-quality result.
13. **Harness patch provenance is reproducible only against the pinned clean
    baseline.** The current external checkout contains divergent dirty changes;
    `scripts/check_combined_harness_patch.py` now archives commit
    `7652836056c59e044f093e3c13ed7438c814169e` into a temporary tree, performs
    the notebook's one-line legacy seed normalization, and passes
    `git apply --check` without changing either checkout.
14. **Path-hazard recovery separates route safety from impossible geometry.**
    The new generator checks that every hidden fatal-path cell leaves an
    alternate public route. Adaptive and two-phase retain 100% held-out wins
    while reducing actions against baseline; the corridor prior exhausts its
    budget on a substantial minority. This is a CPU diagnostic only.
15. **Repeated hazards do not justify immediate fallback promotion.** The
    two-hazard stress matrix keeps adaptive and two-phase at 100% wins, while
    corridor remains budget-sensitive. Immediate adaptive fallback saves
    actions only on the two-hazard case and raises life losses in both the one-
    and two-hazard cases, so the existing conservative adaptive policy remains
    the retained diagnostic candidate.

## Ranked hypotheses

1. **Highest expected value:** obtain an equivalent GPU and reproduce the
   winner profile through one-game preflight → full 25-game audited offline run
   → competition rerun. This is required to turn local edits into score
   evidence and has the clearest measurement path.
2. **High information value, lower immediate score certainty:** use the
   delayed-feedback result to design a real-game observation-timing probe once
   an equivalent GPU is available. The sequence policy is robust in this toy
   harness, but the delayed mechanic does not establish hidden-goal transfer.
3. **Medium value, high risk:** test one isolated memory/context or adaptive
   recovery change against the frozen winner profile under identical budgets.
   Keep the incumbent package untouched and require paired per-game evidence.
4. **Low current priority:** fine-tuning on synthetic games. Revisit only after
   a concrete failure mode, held-out transfer gain, and exact GPU preflight.

## Compute and submission posture

- Local compute is suitable for CPU contracts and probes only. Do not launch a
  full model run on this host.
- A qualifying Kaggle run consumes the notebook's 32,400-second budget. The
  current verified remaining submission count is unknown; query it before use
  from an authenticated Kaggle environment. The `kaggle` binary is not on
  PATH, but the authenticated `uvx --with six --from kaggle` path is available.
- The latest confirmed combined submission is `57910947` at 6.92. The
  protected candidate remains `57385516` at 29.69; historical version 1 at
  `57345516` remains 29.49.
- No watcher, automatic resubmission, or duplicate submission is active.
- Kernel version 1 completed its offline run and competition rerun as code
  submission `57910947`, with public score 6.92. Do not promote it or submit
  the patched local candidate after the deadline; keep the protected incumbent
  and the visible 29.49 historical candidate for Kaggle's automatic final
  selection.

## Exact next actions

1. Keep this file current after every public score, completed artifact audit,
   or material local benchmark; do not rerun completed experiments without a
   confirmation need.
2. For any future GPU work, run the side-effect-free preflight and record the
   exact GPU, package hash, model inputs, and available wall-clock budget.
3. Keep submission `57910947` recorded as terminal 6.92 and rejected for
   promotion; do not treat its offline output page as a competition-mode trace.
4. Keep the 29.69 incumbent protected and do not submit the patched local
   candidate after the closed deadline. If the competition reopens, rerun the
   preflight → full audit → competition sequence with the summary profile still
   disabled by default.
5. If Kaggle later exposes a true competition-mode artifact, audit it separately
   from the recovered offline evidence in
   `build/combined-kaggle-full/audit-version-1-20261102/recovered_output_manifest.json`.
6. Diagnose analyzer timeouts only with an equivalent GPU and a comparable
   competition-mode artifact; the current offline timeouts do not explain the
   public score by themselves.
7. Keep fine-tuning deferred until held-out transfer evidence justifies it.
8. Keep the delayed-feedback and path-hazard matrices as diagnostic regression
   fixtures, including the two-hazard extension and adaptive ablation. Do not
   promote `SequenceExplorer`, adaptive, corridor, or immediate fallback into
   the scored notebook without a held-out real-game transfer result.

## Reproduction commands

```bash
python -m unittest discover -s tests -q
python scripts/build_combined_kaggle_package.py \
  --kernel-id '<kaggle-username>/arc3-m2-combined-profile-final-preflight' \
  --mode preflight --offline-max-games 1 \
  --output-dir build/combined-kaggle-final-preflight-20261103
python scripts/build_combined_kaggle_package.py \
  --kernel-id '<kaggle-username>/arc3-m2-combined-profile-final' \
  --mode full --output-dir build/combined-kaggle-final-20261103
python scripts/arc3_gpu_preflight.py
python scripts/check_combined_harness_patch.py \
  --external-repo external/arc-agi-3-solution \
  --base-commit 7652836056c59e044f093e3c13ed7438c814169e \
  --output build/arc3-harness-patch-check-20261101-repeatable/report.json
uv venv /tmp/arc3-cpu-smoke --python /home/izu/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
uv pip install --python /tmp/arc3-cpu-smoke/bin/python --no-index --find-links arc_agi_3_wheels arc-agi==0.9.8
/tmp/arc3-cpu-smoke/bin/python scripts/arc3_cpu_environment_smoke.py \
  --environment-dir environment_files --max-games 25 --actions-per-game 1 \
  --output build/arc3-cpu-smoke-20261101/report.json
python scripts/arc3_generated_game_probe.py --seeds 0 1 2 3 4
python scripts/arc3_generated_game_probe.py --blind-policy --seeds 101 102
python scripts/arc3_generated_game_probe.py --two-phase-policy --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --corridor-policy --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --two-phase-recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --corridor-recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --budget-curve --seeds 2000 2001 2002
python scripts/arc3_generated_game_probe.py --blind-policy --seeds $(seq 100 199) --max-actions 512 --output /tmp/baseline.json
python scripts/arc3_generated_game_probe.py --two-phase-policy --seeds $(seq 100 199) --max-actions 512 --output /tmp/two-phase.json
python scripts/arc3_generated_game_probe.py --corridor-policy --seeds $(seq 100 199) --max-actions 512 --output /tmp/corridor.json
python scripts/arc3_generated_game_probe.py --corridor-policy --geometry rotate180 --seeds $(seq 100 199) --max-actions 512 --output /tmp/corridor-rotate180.json
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry center_goal --seeds $(seq 100 199) --max-actions 512 --output /tmp/adaptive-center-goal.json
python scripts/arc3_generated_game_probe.py --adaptive-recovery-policy --geometry center_goal --seeds $(seq 100 199) --max-actions 1024 --output /tmp/adaptive-center-goal-recovery.json
python scripts/arc3_generated_game_probe.py --two-phase-policy --geometry near_goal --seeds $(seq 100 199) --max-actions 512 --output /tmp/two-phase-near-goal.json
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry offset_goal --seeds $(seq 100 199) --max-actions 512 --output /tmp/adaptive-offset-goal.json
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry random_goal --seeds $(seq 100 199) --max-actions 512 --output /tmp/adaptive-random-goal.json
python scripts/arc3_generated_game_probe.py --budget-curve --budget-values 128 192 256 320 384 512 --levels 5 --adaptive-policy --geometry random_goal --seeds $(seq 100 199) --output /tmp/adaptive-random-goal-budget.json
python scripts/arc3_generated_game_probe.py --adaptive-recovery-policy --geometry random_goal --seeds $(seq 100 199) --max-actions 1024 --output /tmp/adaptive-random-goal-recovery.json
python scripts/arc3_generated_game_probe.py --sequence-benchmark --geometry random_goal --seeds 101 102 --levels 3 --max-actions 1024
python scripts/arc3_generated_game_probe.py --delayed-feedback-benchmark --geometry random_goal --seeds 101 102 --levels 3 --max-actions 1536
python scripts/arc3_path_hazard_matrix.py \
  --output-dir build/arc3-policy-benchmark-20261101/path_hazard \
  --geometries canonical flip_vertical flip_horizontal rotate180 center_goal near_goal offset_goal random_goal \
  --seed-splits 100-199 200-299 1000-1099 \
  --levels 3 --max-actions 1024
python scripts/arc3_path_hazard_matrix.py \
  --output-dir build/arc3-policy-benchmark-20261101/multi_path_hazard \
  --geometries canonical flip_vertical flip_horizontal rotate180 center_goal near_goal offset_goal random_goal \
  --seed-splits 100-199 200-299 1000-1099 \
  --levels 3 --max-actions 1024 --hazard-count 2
python scripts/arc3_adaptive_recovery_ablation.py \
  --output-dir build/arc3-policy-benchmark-20261101/adaptive_ablation \
  --geometries canonical flip_vertical flip_horizontal rotate180 center_goal near_goal offset_goal random_goal \
  --seed-splits 100-199 200-299 1000-1099 \
  --hazard-counts 1 2 --levels 3 --max-actions 1024
```

For Kaggle package/build/audit commands, use the documented workflow in
`README.md` and `docs/arc-agi-3/generator-and-finetuning-plan.md`. Record the
exact kernel version, artifact path, audit result, submission reference, and
public score here after each authorized remote run.

## Evidence index

- Active candidate and combined score history: `README.md`.
- Low-resource decisions and generator evidence:
  `docs/arc-agi-3/generator-and-finetuning-plan.md`.
- Combined public crawl and transfer cautions:
  `docs/arc-agi-3/milestone-2-public-crawl-2026-10-01.md`.
- Flash-family experiment ledger: `FLASH_EXPERIMENTS_SINCE_3_40.md`.
- Ensemble hypothesis and acceptance gates: `CODEX_ARC_ENSEMBLE_MEMORY_PLAN.md`.
- Machine-readable offline outputs and audits: `build/kaggle-output/`.
- Two-phase policy benchmark records: `build/arc3-policy-benchmark-20261031/`.
- Non-far start/goal audit records:
  `build/arc3-policy-benchmark-20261031/geometry/non_far/`.
- Non-far five-level budget curves:
  `build/arc3-policy-benchmark-20261031/geometry/non_far/budget_curves_100_199.json`.
- Non-far hidden-hazard recovery:
  `build/arc3-policy-benchmark-20261031/geometry/non_far/recovery_summary_100_199.json`.
- Observable sequence-mechanic records and aggregate:
  `build/arc3-policy-benchmark-20261031/sequence/summary.json`.
- Delayed-feedback sequence records and aggregate:
  `build/arc3-policy-benchmark-20261031/delayed_feedback/summary.json`.
- Path-hazard matrix runner and regression tests:
  `scripts/arc3_path_hazard_matrix.py` and
  `tests/test_arc3_path_hazard_matrix.py`.
- Path-hazard matrix records and aggregate (2,400 games, 7,200 levels):
  `build/arc3-policy-benchmark-20261101/path_hazard/summary.json` (SHA-256
  `0b58f62f046a19ed8f0cb942556c8f697506ec683c949232a0b08002c8fd54a4`).
- Two-hazard matrix runner extension and regression coverage:
  `scripts/arc3_path_hazard_matrix.py` and
  `tests/test_arc3_path_hazard_matrix.py`.
- Two-hazard matrix records and aggregate (2,400 games, 7,200 levels):
  `build/arc3-policy-benchmark-20261101/multi_path_hazard/summary.json`
  (SHA-256
  `6e7e279b73a7ee7239315e8cbb39df1eb9d3c7e4b16bc3ec00e6b1bffe43266c`).
- Adaptive recovery ablation runner and regression coverage:
  `scripts/arc3_adaptive_recovery_ablation.py` and
  `tests/test_arc3_adaptive_recovery_ablation.py`.
- Paired adaptive ablation records and aggregate (4,800 games, 14,400 levels):
  `build/arc3-policy-benchmark-20261101/adaptive_ablation/summary.json`
  (SHA-256
  `315f14846d4acd69b4dc029bcd67de2faf95f63e7beb446144b79e360a222b90`).
- Active package builder: `scripts/build_combined_kaggle_package.py` with
  regression checks in `tests/test_build_combined_kaggle_package.py`.
- Generated final active package manifests and hashes:
  `build/combined-kaggle-final-preflight-20261103/build-manifest.json` and
  `build/combined-kaggle-final-20261103/build-manifest.json`.
- Isolated CPU environment smoke report (25/25 resets and actions):
  `build/arc3-cpu-smoke-20261101/report.json` (SHA-256
  `7433e3e96aa8527c29503fdc39332fb709e95fa84fd24941c7fbeb1243548a74`).
- Pinned-baseline harness patch provenance report:
  `build/arc3-harness-patch-check-20261101/report.json` (SHA-256
  `762a961c2828865f29d388d49c7acf5925d04adb1873949af4fdb24caed4d5e6`).
- Repeatable harness patch checker: `scripts/check_combined_harness_patch.py`
  with regression tests in `tests/test_check_combined_harness_patch.py`.
- Repeatable pinned-baseline provenance report:
  `build/arc3-harness-patch-check-20261101-repeatable/report.json` (SHA-256
  `37b2d95cdb37e55cb9f16eb2f9e999f4f0e759e0d303ca6e21bb3b9f5ca2e5f4`).
- Objective requirement audit:
  `build/arc3-optimization-audit-20261101/requirements-audit.json`.
