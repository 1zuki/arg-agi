# ARC-AGI-3 Qwen3.8 + checkpoint-8 + adapter-fix candidate

**Latest experimental handoff:** [Flash-Next/MTP no-overlay c12 Kaggle package](kaggle/flash-next-nostate-c12/README.md).
Includes the full notebook, corrected preflight, and run/audit/submit instructions.
This c12 variant has not completed a corrected preflight or full run; the public **3.40** incumbent remains protected.

The primary submission notebook is `arc-agi.ipynb`. It preserves the public
[Tufa Labs duck harness](https://github.com/Tufalabs/duck-harness) as the solver
base, swaps its local analyzer to `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, requests `xhigh` reasoning, and
keeps the checkpoint after eight actions from a single batched `action(...)`
call, maps engine `ACTION7` to the model-facing `UNDO` action, and synchronizes
the analyzer's state after automatic resets.

This is the primary candidate, not yet a measured score improvement over the
previous checkpoint-8 variant. The repository can validate the patch and
provenance locally; serving the 31 GB FP8 model and completing gameplay require
an RTX Pro 6000 Kaggle Save & Run and then a competition rerun.

Required Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share` (dataset version 4)
- `driessmit1/arc3-vllm-h100-wheelhouse-v3` (dataset version 1)
- `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (dataset version 1)
- the official `arc-prize-2026-arc-agi-3` competition dataset

The old local CWM/RL baseline path has been removed from this snapshot. See
`ARCHITECTURE.md` for the current submission flow and remaining risks.

## Why this candidate

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
explicit experiments and are recorded in the run manifest:

- `TAAF_OFFLINE_GAMES`: comma/space-separated game ids for non-submission
  Save & Run only.
- `TAAF_OFFLINE_MAX_GAMES`: first N offline games for non-submission
  Save & Run only.
- `TAAF_BENCHMARK_CONCURRENCY`: overrides `bm.solver.concurrency`.
- `TAAF_ANALYZER_TIMEOUT`: overrides `bm.solver.analyzer_timeout`.
- `TAAF_MAX_ACTIONS_PER_GAME`: overrides `bm.solver.max_actions_per_game`.
- `TAAF_MAX_RUNTIME_S_PER_GAME`: overrides
  `bm.solver.max_runtime_s_per_game`.

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

## Build, run, and audit on Kaggle

The local builder creates private, no-internet Kaggle kernel packages under the
ignored `build/` directory. Replace the placeholder ids with your authenticated
Kaggle username and distinct slugs:

```bash
python scripts/build_kaggle_package.py \
  --kernel-id '<kaggle-username>/<preflight-kernel-slug>' \
  --mode preflight

python scripts/build_kaggle_package.py \
  --kernel-id '<kaggle-username>/<full-kernel-slug>' \
  --mode full
```

The preflight package injects `TAAF_OFFLINE_MAX_GAMES=1` only for
non-competition runs. The full package is code-equivalent to the source
notebook after output cleanup. Both metadata files request one RTX Pro 6000,
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

The next measured experiment is a one-game preflight with Duck concurrency
below 28 (for example 8, 12, or 16) to test the observed KV-cache pressure
against the vLLM eight-sequence limit. Keep the 3.40 submission as the
incumbent until a variant improves both timeout behavior and score, then run a
fresh full audit before submitting anything else.

Local checks: `python -m unittest discover -s tests -v`. Teardown behavior tests
require the downloaded pinned source under `build/flash-teardown-source`
(the download command is in `tests/test_flash_teardown.py`); otherwise those
fixture-dependent tests explicitly skip. Parquet auditing uses PyArrow in an
isolated `uvx` environment, not a new global or project dependency.
