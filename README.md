# ARC-AGI-3 Qwen3.8 + checkpoint-8 candidate

The primary submission notebook is `arc-agi.ipynb`. It preserves the public
[Tufa Labs duck harness](https://github.com/Tufalabs/duck-harness) as the solver
base, swaps its local analyzer to `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, requests `xhigh` reasoning, and
adds one isolated policy change: a checkpoint after eight actions from a single
batched `action(...)` call.

This is the primary candidate, not a locally proven score improvement. The
repository can validate the patch and provenance locally; serving the 31 GB
FP8 model and completing gameplay require an RTX Pro 6000 Kaggle Save & Run and
then a competition rerun.

Required Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share` (dataset version 4)
- `driessmit1/arc3-vllm-h100-wheelhouse-v3` (dataset version 1)
- `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (dataset version 1)
- the official `arc-prize-2026-arc-agi-3` competition dataset

The old local CWM/RL baseline path has been removed from this snapshot. See
`ARCHITECTURE.md` for the current submission flow and remaining risks.

## Why this candidate

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
`/kaggle/working`, applies the exact 19-line checkpoint-8 source change and
Qwen3.8 setup substitutions, verifies the post-patch hashes, and only then
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

No Kaggle upload, run, competition rerun, or submission has been performed from
this repository preparation workflow.

## Verification gate

Before treating this as a submission improvement:

1. Run and pass the generated one-game preflight package.
2. Run and pass the full package on all 25 offline public games.
3. Only after both audited gates pass, run the full package as a competition
   rerun and require a passing `--mode submission` audit.

Do not claim the external `2.03` or public-game scores as this notebook's result
until those gates pass.
