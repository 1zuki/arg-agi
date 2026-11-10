# ARC-AGI-3 candidate repository

## Active candidate: combined Milestone 2 profile

The notebook being improved in this work is
[`arc-agi-3-milestone-2-solution-combined.ipynb`](arc-agi-3-milestone-2-solution-combined.ipynb).
It starts from Daniel Franzen's Milestone 2 winner profile and keeps its
Pennyroyal/SGLang serving path while testing compatible configuration ideas from
other public solutions.

Verified Kaggle submission history for this kernel family:

| Kernel version | Kaggle submission | Public score |
| --- | ---: | ---: |
| 1 | `57345516` | **29.49** |
| 2 | `57321574` | 22.62 |
| 3 | `57385516` | **29.69** |
| 4 | `57414229` | 22.80 |
| 6 | `57472977` | 29.43 |
| 7 | `57514838` | 0.00 |
| 8 | `57587807` | 0.00 |
| 13 | `57747556` | 8.98 |
| full v1 | `57910947` | 6.92 |

These are historical combined-profile results, not scores produced by the
current local checkout. The protected incumbent remains Version 3 at 29.69;
the separate full-kernel rerun `57910947` completed at 6.92 and is rejected for
promotion. Its saved Kaggle output is an offline diagnostic artifact, so the
hidden competition rerun cannot be attributed to a specific runtime change.
The local candidate keeps the rolling-summary path opt-in and is prepared for a
future equivalent-GPU run, but is not a submitted replacement.

## Separate Flash-Next candidate family

The **3.40** protected result (submission `56170646`) and the **2.32** ACTION7
result (submission `56369560`) belong to the separate Flash-Next/MTP c28 family:
[Flash ACTION7 passthrough, c28](kaggle/flash-next-action7/README.md).
They should not be used as scores for the combined Milestone 2 notebook.
The c12/c8 variants did not beat 3.40; see `FLASH_EXPERIMENTS_SINCE_3_40.md`.

## Current testing boundary

The combined Milestone 2 notebook is a GPU-backed scored-run artifact. Local
CPU checks cover notebook JSON, runnable-cell syntax, patch/package contracts,
and output auditing; they cannot validate Qwen serving, VRAM fit, gameplay, or
leaderboard score. The active profile needs an RTX PRO 6000-class Kaggle run,
with a 32,400-second wall-clock budget. Offline mode schedules 25 games at a
25-minute per-game limit; competition mode uses the gateway game list and a
different schedule. Historical combined-profile public scores varied widely,
so the CPU suite is a regression check, not a score forecast.

Run the side-effect-free hardware gate before preparing a GPU package:

```bash
python scripts/arc3_gpu_preflight.py
```

Exit code `2` means the current host does not meet the planning threshold. It
does not prove that a qualifying GPU will finish or score well; the exact
one-game serving preflight remains required.

The low-resource plan is recorded in
[`docs/arc-agi-3/generator-and-finetuning-plan.md`](docs/arc-agi-3/generator-and-finetuning-plan.md).
Generated-game probes remain the active CPU-scale path. Fine-tuning stays
deferred until a held-out transfer test and equivalent GPU access exist.

## Active combined package workflow

The active notebook has a dedicated local package builder. It validates the
current notebook anchors and runnable cells, emits exact model and dataset
metadata, and writes a `build-manifest.json` with source and generated SHA-256
hashes. It never authenticates, uploads, runs, or submits to Kaggle:

```bash
python scripts/build_combined_kaggle_package.py \
  --kernel-id '<kaggle-username>/arc3-m2-combined-profile-final-preflight' \
  --mode preflight --offline-max-games 1 \
  --output-dir build/combined-kaggle-final-preflight-20261103

python scripts/build_combined_kaggle_package.py \
  --kernel-id '<kaggle-username>/arc3-m2-combined-profile-final' \
  --mode full --output-dir build/combined-kaggle-final-20261103
```

The preflight package limits the offline game list at the active
`bm.games = _offline_games(...)` anchor (one game by default). Competition
reruns use the gateway branch and are unaffected. The full package is
code-equivalent to the source notebook after execution-output cleanup. The
generated artifacts are `build/combined-kaggle-final-preflight-20261103/` and
`build/combined-kaggle-final-20261103/`.

After `bm.run` completes, the notebook validates offline `benchmark.json`
records, terminal game states, finite scores, action-history lengths, duplicate
game IDs, and the Parquet framing/schema. The offline scaffold must contain zero
rows; a competition run must contain at least one row with the exact
`row_id`, `game_id`, `end_of_game`, and `score` columns. Row IDs and game IDs
must be non-empty and unique by row, terminal flags must be boolean, and scores
must be finite and non-negative. It writes aggregate metadata to
`combined_run_contract.json`; competition mode omits per-game details from that
contract. A run without a valid benchmark or output fails closed.

After reviewing the packages and completing the local gates, the user-run
Kaggle commands are:

```bash
uvx --from kaggle kaggle kernels push -p build/combined-kaggle-final-preflight-20261103
uvx --from kaggle kaggle kernels push -p build/combined-kaggle-final-20261103
```

The bounded generator probe is runnable without the model or a GPU:

```bash
python scripts/arc3_generated_game_probe.py --seeds 0 1 2 3 4
python scripts/arc3_generated_game_probe.py --blind-policy --seeds 101 102
python scripts/arc3_generated_game_probe.py --two-phase-policy --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --corridor-policy --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --corridor-policy --geometry rotate180 --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry center_goal --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --two-phase-policy --geometry near_goal --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --adaptive-policy --geometry random_goal --seeds 101 102 --max-actions 512
python scripts/arc3_generated_game_probe.py --budget-curve --budget-values 128 192 256 320 384 512 --levels 5 --adaptive-policy --geometry random_goal --seeds 101 102
python scripts/arc3_generated_game_probe.py --recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --two-phase-recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --corridor-recovery-policy --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --adaptive-recovery-policy --geometry center_goal --seeds 101 102 --max-actions 1024
python scripts/arc3_generated_game_probe.py --sequence-benchmark --geometry random_goal --seeds 101 102 --levels 3 --max-actions 1024
python scripts/arc3_generated_game_probe.py --delayed-feedback-benchmark --geometry random_goal --seeds 101 102 --levels 3 --max-actions 1536
python scripts/arc3_path_hazard_matrix.py --output-dir build/arc3-policy-benchmark-20261101/path_hazard
python scripts/arc3_generated_game_probe.py --budget-curve --seeds 2000 2001 2002
python -m unittest tests.test_arc3_generated_game_probe -v
```

The oracle result is a reproducibility and state-contract check. The blind
policy run is a small held-out exploration regression, not an agent score or a
forecast for the hidden competition set.

`--two-phase-policy` is an explicit CPU experiment. `TwoPhaseExplorer` first
walks the public grid without spending `TOGGLE` actions, then probes and
revisits observed cells only when movement did not finish the level. It is
kept separate from `BlindExplorer` until a real ARC-AGI-3 run shows transfer;
the paired seed-split records are in
`build/arc3-policy-benchmark-20261031/summary.json`.

`--corridor-policy` is a second explicit CPU experiment. `PublicCorridorExplorer`
uses only the observed grid and start position to choose a geometrically distant
target, then probes the union of shortest public-grid corridors before returning
to that target. It reached 300/300 games at 512 actions on seeds 100–199,
200–299, and 1000–1099, with mean actions 100.603/100.257/99.530. The
geometric prior is specific to this generated probe, so it is not a replacement
for `BlindExplorer` and does not imply ARC-AGI-3 transfer. Its full paired
records, recovery run, and budget curve are in the same benchmark directory.

`--adaptive-policy` wraps the corridor fast path with a `TwoPhaseExplorer`
fallback. On a synthetic `center_goal` layout where the visible board extends
beyond a hidden center goal, corridor alone reached 115/300 games while the
adaptive policy reached 300/300 at 512 actions. This fallback result is still
CPU-only generated-game evidence.

The probe also supports
`--geometry canonical|flip_vertical|flip_horizontal|rotate180|center_goal|near_goal|offset_goal|random_goal`.
On seeds 100–199, the corridor candidate stayed at 300/300 wins for all three
alternate orientations, while the frozen blind baseline fell to 251/300,
270/300, and 224/300 respectively. These transforms are a robustness check for
public-grid orientation, not evidence about arbitrary ARC-AGI-3 goal geometry.

The `near_goal`, `offset_goal`, and `random_goal` layouts remove the far-side
goal prior. Across three held-out seed splits, two-phase and adaptive each
completed 300/300 games at 512 actions for all three layouts. The corridor
candidate fell to 125/300, 151/300, and 38/300 on seeds 100–199, so its fast
path remains a diagnostic and is not promoted into the baseline. Detailed
records and five-level budget curves are in
`build/arc3-policy-benchmark-20261031/geometry/non_far/`.
The hidden-hazard recovery report in the same directory keeps two-phase and
adaptive at 100/100 on all three layouts; corridor recovery is substantially
less reliable on the random-goal layout.

`--sequence-benchmark` adds a separate observable multi-step mechanic. Each
sequence level reports progress publicly, but hides the ordered interaction
cells and the total count. Across eight geometries and three held-out seed
splits (2,400 games, 7,200 levels, 1,024 actions), `SequenceExplorer` completed
2,400/2,400 games with no budget exhaustion. The baseline completed 1,463,
adaptive 1,061, two-phase 21, and corridor 0. This is a CPU-only generated-game
diagnostic about the value of observable progress feedback; it is not a scored
notebook change or an ARC-AGI-3 transfer claim. The per-split records and
aggregate are in `build/arc3-policy-benchmark-20261031/sequence/summary.json`.

`--delayed-feedback-benchmark` adds a second multi-step mechanic in which
ordered interactions unlock immediately but public progress arrives after a
deterministic delay with occasional one-step noise. Across the same eight
geometries and three held-out seed splits (2,400 games, 7,200 levels, 1,536
actions), `SequenceExplorer` completed 2,400/2,400 games with no budget
exhaustion. The baseline completed 1,449, adaptive 1,022, two-phase 14, and
corridor 1. This result measures tolerance to stale feedback inside the local
harness; it is diagnostic only and does not change the scored notebook. The
per-split records and aggregate are in
`build/arc3-policy-benchmark-20261031/delayed_feedback/summary.json`.

`scripts/arc3_path_hazard_matrix.py` adds a stricter recovery diagnostic. Each
level hides one hazard on the nominal route but guarantees an alternate public
route, then compares baseline, two-phase, corridor, and adaptive policies. The
matrix covers eight geometries, three held-out seed splits, 2,400 games, and
7,200 levels at 1,024 actions. Adaptive and two-phase completed 2,400/2,400
games with mean actions 50.093 and 57.502; baseline also completed 2,400/2,400
but used 100.889 mean actions. Corridor completed 1,879/2,400 and exhausted
its budget on 521 games. This is generated-game recovery evidence only; it does
not change the scored notebook or establish ARC-AGI-3 transfer. Reports are in
`build/arc3-policy-benchmark-20261101/path_hazard/`.

The same runner accepts `--hazard-count 2` for a repeated-failure stress test.
The two-hazard matrix preserves an alternate public route on all 7,200 levels
and covers the same 2,400 games. Adaptive and two-phase retained 2,400/2,400
wins; baseline also retained 2,400/2,400; corridor reached 1,857/2,400 and
exhausted its budget on 543 games. This is still generated-game CPU evidence.
The aggregate is in
`build/arc3-policy-benchmark-20261101/multi_path_hazard/summary.json`.

`scripts/arc3_adaptive_recovery_ablation.py` records a paired immediate-fallback
variant over one- and two-hazard cases. It lowers mean actions only in the
two-hazard case while increasing life losses in both cases, so it remains a
rejected diagnostic variant rather than a scored-notebook change. Its aggregate
is `build/arc3-policy-benchmark-20261101/adaptive_ablation/summary.json`.

The recovery probe adds hidden hazards that are omitted from the public board.
It measures whether an observation-only policy records an observable life-loss,
resets, and avoids the failed transition on the next attempt. This is a
recovery/counterfactual regression, not a leaderboard or fine-tuning result.

The bundled public environments can also be checked without the Qwen model.
Create an isolated Python 3.12 environment, install `arc-agi` from the local
wheelhouse, then run:

```bash
uv venv /tmp/arc3-cpu-smoke --python 3.12
uv pip install --python /tmp/arc3-cpu-smoke/bin/python --no-index \
  --find-links arc_agi_3_wheels arc-agi==0.9.8
/tmp/arc3-cpu-smoke/bin/python scripts/arc3_cpu_environment_smoke.py \
  --environment-dir environment_files --output /tmp/arc3-cpu-smoke.json
```

This checks all 25 bundled games, reset handling, available-action reporting,
and one schema-valid action per game. It is an environment compatibility smoke,
not a solver or score evaluation.

If the read-only Skyfall shelf clone is present at `/tmp/arc-agi-3-games-research`,
the packaged trajectories can be replayed on their authored seeds without
calling the generator or an agent API:

```bash
/tmp/arc3-cpu-smoke/bin/python scripts/arc3_skyfall_shelf_smoke.py \
  --shelf-root /tmp/arc-agi-3-games-research \
  --output /tmp/arc3-skyfall-shelf.json
```

This verifies the shelf manifests, 64x64 trajectory observations, action
schemas, and canonical replay completion. It is generated-game/package evidence,
not evidence for hidden-game transfer, fine-tuning benefit, or leaderboard score.

The older checkpoint-8 candidate is `arc-agi.ipynb`. It preserves the public
[Tufa Labs duck harness](https://github.com/Tufalabs/duck-harness) as the solver
base, swaps its local analyzer to `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, requests `xhigh` reasoning, and
keeps the checkpoint after eight actions from a single batched `action(...)`
call, maps engine `ACTION7` to the model-facing `UNDO` action, and synchronizes
the analyzer's state after automatic resets.

This older checkpoint-8 candidate is retained for lineage comparison. The
repository can validate its patch and provenance locally; serving the 31 GB FP8
model and completing gameplay require an RTX Pro 6000 Kaggle Save & Run and
then a competition rerun.

Required Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share` (dataset version 4)
- `driessmit1/arc3-vllm-h100-wheelhouse-v3` (dataset version 1)
- `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (dataset version 1)
- the official `arc-prize-2026-arc-agi-3` competition dataset

The old local CWM/RL baseline path has been removed from this snapshot. See
`ARCHITECTURE.md` for the current submission flow and remaining risks.

## Attribution

- **Tufa Labs and Jeroen Cottaar**: original Duck harness, solver, and
  competition notebook: [Tufa Labs' Duck harness](https://github.com/Tufalabs/duck-harness).
- **Daniel Franzen**: Milestone 2 solution and public fork at
  [`da-fr/arc-agi-3-solution`](https://github.com/da-fr/arc-agi-3-solution),
  which is the foundation for the Milestone 2 material carried in this
  repository.

Please preserve both attributions when reusing the corresponding harness or
Milestone 2 solution work.

## Older checkpoint-8 lineage (not the active combined profile)

The previous repository candidate completed an audited 25-public-game Kaggle
run on 2026-09-08 with a mean score of `4.671335090995715`, 27 levels cleared,
and positive score on 18 of 25 games. Trace analysis exposed two deterministic
adapter defects: `ACTION7` was advertised but rejected by the reverse action
mapping, and automatic reset could leave stale terminal context in the analyzer.
The new candidate changes only those defects while preserving the measured
model, sampling, checkpoint, and runtime settings. Its score remains unverified
until it is rerun.

Public external evidence in
[sonpham-org/arc-3](https://github.com/sonpham-org/arc-3/tree/3ba35ce5a9a81efc068f0b40db1b1e92c785d588)
reports the following on the 25 public games:

- checkpoint-8 Qwen3.8 replicas: `3.8361` and `5.8142` mean score
  (`4.8251` pair mean), plus an exact third reproduction at `4.4358`;
- published no-checkpoint controls using the Kaggle 4x visual shape: `3.5063`
  and `5.3877` (`4.4470` pair mean);
- a Kaggle public score of `2.03` for submission `55551321` using the same
  Qwen3.8/xhigh/checkpoint-8 shape.

These are external measurements, not results produced from this repository.
The source artifacts are the public
[checkpoint-8 manifest](https://github.com/sonpham-org/arc-3/blob/3ba35ce5a9a81efc068f0b40db1b1e92c785d588/harnesses/taaf-plain-checkpoint8/RESTORED_CHAMPION.json),
[run index](https://github.com/sonpham-org/arc-3/blob/3ba35ce5a9a81efc068f0b40db1b1e92c785d588/docs/data/runs-index.json),
and
[Kaggle-control manifest](https://github.com/sonpham-org/arc-3/blob/3ba35ce5a9a81efc068f0b40db1b1e92c785d588/harnesses/taaf-kaggle203-nocap-control/MANIFEST.md).
The model snapshot is also public on
[Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)
and
[Kaggle](https://www.kaggle.com/datasets/jakobbrggen/qwen3-8-27b-fp8-hf-snapshot).

The notebook keeps the measured runtime shape: one pass, concurrency 28,
900-second analyzer timeout, 7,920 seconds per game, 65,536-token server
context, 32,768-token analyzer context, sampling `0.6 / 0.95 / 20`, current-grid
images at 4x, and the setup smoke test before gameplay. The exact pinned TAAF
deployment payload declares a 32,400-second (nine-hour) wall-clock budget. The
launcher sets a global soft deadline 30 minutes before that limit so an abnormal
long rerun can close its scorecard and run teardown instead of being hard-killed;
normal scored-shape runs retain the full 7,920-second per-game policy.

## Fail-closed preparation

Kaggle inputs are read-only. The launcher therefore verifies the complete
73-file TAAF tree digest plus critical source and benchmark-pickle hashes,
copies the 1.7 MB bundle to
`/kaggle/working`, applies the checkpoint-8 and adapter fixes plus Qwen3.8 setup
substitutions, verifies the post-patch hashes, and only then
updates `sys.path` and unpickles the benchmark.
It also verifies the model metadata hashes, FP8 architecture, xhigh-capable chat
template, exact 80-file inventory, and all 77 payload CRCs (including 66 weight
shards). The dataset's `crc32.txt` has three stale metadata entries; the launcher
accepts their corrected CRCs only after those files match the immutable model
revision by pinned SHA-256. Before setup, it also pins the vLLM wheelhouse metadata and
verifies all 178 published payload files against its pinned `SHA256SUMS`
inventory. A mismatch stops the run.

## Launcher knobs

Defaults preserve the candidate above. Optional environment overrides are for
explicit non-submission experiments and are printed with the resolved solver
configuration:

- `TAAF_OFFLINE_GAMES`: comma/space-separated exact game ids for non-submission
  Save & Run only; unknown ids fail closed.
- `TAAF_OFFLINE_MAX_GAMES`: first N selected offline games for non-submission
  Save & Run only; it must be a positive integer.
- `TAAF_BENCHMARK_CONCURRENCY`: positive-integer override for
  `bm.solver.concurrency` during non-submission runs.
- `TAAF_ANALYZER_TIMEOUT`: positive-finite override for
  `bm.solver.analyzer_timeout` during non-submission runs.
- `TAAF_MAX_ACTIONS_PER_GAME`: positive-integer override for
  `bm.solver.max_actions_per_game` during non-submission runs.
- `TAAF_MAX_RUNTIME_S_PER_GAME`: positive-finite override for
  `bm.solver.max_runtime_s_per_game` during non-submission runs; it cannot
  exceed the notebook wall-clock budget. Competition settings remain fixed.

## Run manifest

The launcher writes `/kaggle/working/taaf_run_manifest.json` and a short
`/kaggle/working/taaf_run_summary.txt` while it starts up. These files are for
debugging Kaggle reruns and include only non-secret fields: variant and model
identity, source and patched hashes, model-validation status, submission mode,
Python/platform info, known safe TAAF/Kaggle environment knobs, resolved input
paths, solver attributes, the pinned wall-clock/soft-deadline contract, and
offline selected game ids for non-submission runs.
Every successful run ends with `benchmark_completed` / `completed`. Submission
mode keeps TAAF's hidden-run diagnostics boundary intact: it suppresses
`benchmark.json`, frame sidecars, and `diagnostics.html`, and writes only a
compact `taaf_submission_health.json` with aggregate run counts and no game ids,
scores, actions, frames, or traces.

## Older checkpoint-8 build, run, and audit on Kaggle

The older checkpoint-8 builder creates private, no-internet Kaggle kernel
packages under the ignored `build/` directory. It targets `arc-agi.ipynb`, not
the active combined notebook. Replace the placeholder ids with your
authenticated Kaggle username and distinct slugs:

```bash
python scripts/build_kaggle_package.py \
  --kernel-id '<kaggle-username>/<preflight-kernel-slug>' \
  --mode preflight

python scripts/build_kaggle_package.py \
  --kernel-id '<kaggle-username>/<full-kernel-slug>' \
  --mode full
```

The preflight package injects a one-game slice at the offline game-list anchor
only for non-competition runs. Explicit Save & Run sessions can use the
`TAAF_OFFLINE_*` controls above. The full package is code-equivalent to the
source notebook after output cleanup. Both metadata files request one RTX Pro 6000,
disable internet, attach the pinned datasets and competition, and keep the
kernel private.

Kaggle CLI authentication is required before any remote command. Use Kaggle's
OAuth flow with `uvx --from kaggle kaggle auth login` (or configure a token by
Kaggle's documented alternative), then confirm it with the read-only command
`uvx --from kaggle kaggle kernels list --mine`. The builder does not
authenticate, upload, run, submit, or infer the account name. After reviewing
each generated package, the user-run launch commands are:

```bash
uvx --from kaggle kaggle kernels push -p build/kaggle-preflight
uvx --from kaggle kaggle kernels push -p build/kaggle-full
```

Wait for the exact kernel version to finish successfully before downloading its
outputs. Supplying an explicit version avoids auditing a later rerun by mistake:

```bash
uvx --from kaggle kaggle kernels status \
  '<kaggle-username>/<kernel-slug>'

uvx --from kaggle kaggle kernels output \
  '<kaggle-username>/<kernel-slug>/<version>' \
  -p 'build/kaggle-output/<run-name>'

python scripts/audit_kaggle_output.py \
  --mode preflight \
  'build/kaggle-output/<run-name>'
```

Use `--mode full-offline` for the full 25-public-game Save & Run and
`--mode submission` for a real competition rerun. The auditor is read-only and
prints one JSON report; exit code 0 means every required provenance, runtime,
benchmark, Parquet, server-log, and diagnostics check passed. Non-default solver
experiments fail by default. `--allow-solver-overrides` accepts only positive
overrides that also match the recorded `TAAF_*` manifest values, and still
reports them as warnings.

The previous checkpoint-8 candidate has completed one full public-game Kaggle
run. No Kaggle upload, run, competition rerun, or submission has been performed
for the new `undo-sync` candidate from this workflow.

## Verification gate

Before treating this as a submission improvement:

1. Run and pass the generated one-game preflight package.
2. Run and pass the full package on all 25 offline public games.
3. Only after both audited gates pass, run the full package as a competition
   rerun and require a passing `--mode submission` audit.

Do not claim the external `2.03` or the prior candidate's `4.6713` public-game
mean as this new notebook's result until those gates pass.

## Isolated Flash-Next/MTP candidate (2026-09-11)

`arc-agi-flash-next-mtp.ipynb` and `scripts/build_flash_next_package.py`
package a separate candidate based on Keith Tyser's public serving stack;
they do not replace the checkpoint-8 incumbent (recorded public score `1.31`).
The full candidate completed its competition rerun as kernel version 1, and
submission `56170646` received a public score of **3.40**, above the recorded
1.31 checkpoint-8 incumbent. The local 25-game mean
`5.502654474314745` is offline-only and is not a leaderboard score.

- Preflight `izukia/arc-agi3-flash-next-mtp-preflight/6` completed and passed
  `scripts/audit_flash_output.py`. Output: `build/kaggle-output/flash-preflight-v6`.
  The local package folder is `build/kaggle-flash-preflight-v7`; its name is
  not the Kaggle version number.
- The preflight produced 1,016 actions and score `0` on `tn36-ef4dde99`.
  Its 1,800-second soft deadline is measured from notebook start (including
  setup), not 30 minutes of gameplay. Deadline cancellation is acceptable
  only for the preflight smoke test; the full audit rejects cancelled runs.
- The pinned teardown patch captures metrics once, waits for delayed GPU
  release, and retains the original process-identity, port, GPU-query,
  metric-preservation, and artifact-preservation gates. Preflight observed
  `shutdown_ok=true`, zero survivors, and no watchdog restarts.
- Full kernel `izukia/arc-agi3-flash-next-mtp-full/1` completed from
  `build/kaggle-flash-full-20260911`. Its notebook SHA-256 is
  `6a141753f142ed0d683f34f9acb1ac732a6e333a193b2ad8612d66125e64d9d1`.
  It retained the original 25-game order, 7,920 seconds per game,
  concurrency 28, and nine-hour target budget. The strict offline audit passed
  with 25/25 games, 4,683 actions, clean teardown, and zero watchdog restarts.

Commands for the full-run gate (use a fresh download directory):

```bash
uvx --from kaggle kaggle kernels status izukia/arc-agi3-flash-next-mtp-full/1
uvx --from kaggle kaggle kernels output izukia/arc-agi3-flash-next-mtp-full/1 \
  -p build/kaggle-output/flash-full-v1
uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode full-offline build/kaggle-output/flash-full-v1
```

The exact version was submitted only after completion, offline audit, and score
review. The command used was:

```bash
uvx --from kaggle kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k izukia/arc-agi3-flash-next-mtp-full -v 1 -f submission.parquet \
  -m "Flash-Next MTP; audited public25; GPU-release teardown fix"
```

Do not submit the local offline placeholder as predictions. Offline mean and
public leaderboard score are different measurements. The public stack's fast
setup verifies model/runtime identity metadata, not every payload hash; the
Flash audit reports this limitation rather than claiming full verification.

Post-baseline c8/c12, state-overlay, and ACTION7 experiments did not beat public
3.40. The ACTION7 full offline mean was **5.9754**, but its public submission
`56369560` scored **2.32**. See the [ACTION7 handoff](kaggle/flash-next-action7/README.md)
for the exact artifact hashes and audit commands. Keep 3.40 until a fresh
completed/audited candidate earns a strictly higher public score.

Local checks: `python -m unittest discover -s tests -v`. Teardown behavior tests
require the downloaded pinned source under `build/flash-teardown-source`
(the download command is in `tests/test_flash_teardown.py`); otherwise those
fixture-dependent tests explicitly skip. Parquet auditing uses PyArrow in an
isolated `uvx` environment, not a new global or project dependency.
