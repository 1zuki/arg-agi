# ARC-AGI-3 Tufa Submission

The primary submission notebook is `arc-agi.ipynb`. It is the Tufa Labs duck
harness launcher and expects the solver/runtime artifacts to be attached as
Kaggle datasets.

Required Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share`
- `driessmit1/arc3-vllm-h100-wheelhouse-v3`
- `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
- the official `arc-prize-2026-arc-agi-3` competition dataset

The old local CWM/RL baseline path has been removed from this snapshot. See
`ARCHITECTURE.md` for the current submission flow and remaining risks.

## Launcher knobs

Defaults preserve the real competition rerun behavior. Optional environment
overrides:

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
debugging Kaggle reruns and include only non-secret fields: submission mode,
Python/platform info, known safe TAAF/Kaggle environment knobs, attached dataset
refs and resolved input paths, bundle path, solver override attributes, and
offline selected game ids for non-submission runs.
