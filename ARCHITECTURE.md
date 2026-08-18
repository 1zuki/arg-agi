# ARC-AGI-3 Submission Architecture

This repository now uses the Tufa Labs duck harness notebook as the primary
ARC-AGI-3 submission path. The root notebook, `arc-agi.ipynb`, is a Kaggle
launcher for the TAAF source bundle and benchmark artifacts attached as Kaggle
datasets.

The previous local CWM/RL baseline files have been removed from this repo
snapshot. The submission is no longer generated from `_build_nb.py`, and it no
longer relies on the in-repo `my_agent.py`, `cwm.py`, LS20 planners, or local
world-model checkpoints.

## Primary Notebook

`arc-agi.ipynb` is copied from the Tufa Labs duck harness milestone notebook. It
performs the following steps:

1. Detects whether Kaggle is running a real competition rerun by checking
   `KAGGLE_IS_COMPETITION_RERUN`.
2. Installs the official ARC runtime from the competition wheelhouse at
   `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels`.
3. Finds the attached TAAF source dataset by locating
   `taaf-kaggle-bundle.json` under `/kaggle/input`.
4. Adds the bundled TAAF source repositories to `sys.path` and writes a `.pth`
   file so child processes can import them.
5. Runs the bundle's `setup_commands.json`. These commands are responsible for
   preparing the solver runtime, including the local vLLM/Qwen FP8 analyzer
   server and any wheel/model setup required by the bundle.
6. Unpickles the deployed target and benchmark from `deploy_target.pkl` and
   `benchmark_initial.pkl` in the source bundle.
7. Runs the benchmark and writes outputs under `/kaggle/working`.

The notebook contains infrastructure and diagnostics only. The actual solver
implementation lives in the attached TAAF source dataset, not in this repository.

## Required Kaggle Inputs

The notebook expects these attached Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share`
- `driessmit1/arc3-vllm-h100-wheelhouse-v3`
- `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
- the competition dataset mounted at
  `/kaggle/input/competitions/arc-prize-2026-arc-agi-3`

Kaggle may mount datasets either at `/kaggle/input/<slug>` or at
`/kaggle/input/datasets/<owner>/<slug>`. The notebook records the resolved paths
in `TAAF_KAGGLE_INPUT_PATHS` for the setup commands and solver.

## Submission and Offline Behavior

In a real competition rerun, `TRUE_SUBMISSION` is true. The notebook minimizes
diagnostics, waits for the Kaggle gateway, builds the live competition game list,
and runs against the competition Arcade.

In an interactive Save & Run, `TRUE_SUBMISSION` is false. The notebook uses the
competition dataset's bundled `environment_files` offline, keeps diagnostics
enabled, and creates a stub `submission.parquet` so Kaggle has an output file
even though the offline run is not scored.

Both modes write run artifacts to `/kaggle/working`. During startup the launcher
also refreshes `/kaggle/working/taaf_run_manifest.json` and
`/kaggle/working/taaf_run_summary.txt` with non-secret run metadata: submission
mode, Python/platform info, known safe TAAF/Kaggle environment knobs, attached
input refs and resolved paths, bundle path, selected solver attributes, and
offline selected game ids for non-submission runs. Non-submission runs may also
render `diagnostics.html` inline.

The launcher supports a few environment knobs for iteration. `TAAF_OFFLINE_GAMES`
and `TAAF_OFFLINE_MAX_GAMES` only apply to non-submission Save & Run offline
games. `TAAF_BENCHMARK_CONCURRENCY`, `TAAF_ANALYZER_TIMEOUT`,
`TAAF_MAX_ACTIONS_PER_GAME`, and `TAAF_MAX_RUNTIME_S_PER_GAME` override the
matching `bm.solver` attributes after the benchmark is loaded. Invalid values
raise clear launcher errors instead of silently changing behavior.

## Preserved Runtime Assets

The broad official/runtime assets remain in this repo for now:

- `ARC-AGI-3-Agents/`
- `arc_agi_3_wheels/`
- `environment_files/`
- `.gitignore`, `LICENSE`, and `Dockerfile`

These are kept because they are official/runtime support files or useful local
reference material, and they are not part of the removed CWM/RL baseline.

## Risks and Caveats

- GPU and dataset dependency: the notebook depends on the attached TAAF datasets,
  the FP8 Qwen snapshot, the vLLM wheelhouse, and the correct Kaggle GPU selection
  noted by the original notebook.
- Timeout risk: setup starts local analyzer infrastructure and then runs the
  benchmark. Long startup, gateway delay, or slow games can still consume the
  notebook budget.
- Solver opacity in this repo: the actual solver code is in the attached dataset.
  Reviewing only this repository does not review the solver logic.
- Attribution and baseline caveat: this path uses the public Tufa Labs duck
  harness/milestone submission approach. Any use should preserve attribution and
  account for competition rules and public-baseline expectations.
- Reproducibility: this repository alone is not a complete reproducible
  submission. The exact Kaggle attached datasets and their versions are part of
  the runtime.
