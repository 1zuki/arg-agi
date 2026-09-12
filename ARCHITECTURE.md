# ARC-AGI-3 Submission Architecture

This repository uses the public Tufa Labs duck harness as its ARC-AGI-3 solver
base. The root `arc-agi.ipynb` is a Kaggle launcher for a narrowly modified
candidate: Qwen3.8 with xhigh reasoning, an eight-action batch checkpoint, and
two adapter correctness fixes. Engine `ACTION7` is exposed as `UNDO`, and a
host-issued automatic reset now synchronizes the analyzer before the next turn.
The candidate is intentionally close to upstream TAAF; it does not add a new
planner, prompt replacement, state graph, or other handcrafted solver system.

The previous local CWM/RL baseline files have been removed from this repo
snapshot. The submission is no longer generated from `_build_nb.py`, and it no
longer relies on the in-repo `my_agent.py`, `cwm.py`, LS20 planners, or local
world-model checkpoints.

## Primary Notebook

`arc-agi.ipynb` started from the Tufa Labs duck harness milestone notebook. It
performs the following steps:

1. Detects whether Kaggle is running a real competition rerun by checking
   `KAGGLE_IS_COMPETITION_RERUN`.
2. Installs the official ARC runtime from the competition wheelhouse at
   `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels`.
3. Finds the attached TAAF source dataset by its `taaf-kaggle-bundle.json`
   marker and resolves all attached Kaggle input paths.
4. Verifies a deterministic SHA-256 digest of the complete 73-file bundle plus
   exact hashes for both benchmark pickle payloads, setup and teardown
   artifacts, bundle metadata, `action_names.py`, and the two source files that
   checkpoint-8 modifies.
5. Validates all 180 wheelhouse files, including 178 payload hashes from its
   pinned `SHA256SUMS`, and validates the Qwen3.8 snapshot's exact 80-file
   inventory and all 77 payload CRCs, including 66 weight shards. Three stale
   metadata entries in the dataset's `crc32.txt` are corrected only after the
   mounted files match the immutable Hugging Face revision by pinned SHA-256.
6. Copies the read-only TAAF bundle to
   `/kaggle/working/taaf-qwen38-checkpoint8-undo-sync-bundle`.
7. Applies the checkpoint-8 logic and automatic-reset synchronization to
   `framework/solver.py`/`agent/tool_agent.py`, maps `ACTION7` to `UNDO` in
   `action_names.py`, and substitutes Qwen3.8/xhigh into `setup_commands.json`.
8. Verifies the complete post-patch tree digest and critical file hashes. A
   missing, extra, or changed file, duplicate patch anchor, unexpected model
   snapshot, or incomplete shard set aborts the run before imports or
   unpickling.
9. Adds only the verified writable source to `sys.path` and a `.pth` file, then
   runs its setup command. Setup installs the pinned wheelhouse, checks the GPU,
   starts vLLM, and requires a real-model API smoke test.
10. Unpickles `deploy_target.pkl` and `benchmark_initial.pkl` from the verified
    writable copy, applies the candidate runtime defaults, and then applies any
    deliberate environment overrides.
11. Runs one benchmark pass and writes outputs under `/kaggle/working`. A real
    submission keeps periodic/full diagnostics disabled and emits only a
    compact aggregate health summary for post-run completion auditing.

The notebook contains the patching and launch infrastructure. Tufa's solver
implementation remains in the attached source snapshot and is copied rather
than vendored into this repository. Attribution to Tufa Labs and the original
authors is preserved in the notebook.

## Candidate Contract

The candidate identity is:

- variant: `taaf-qwen38-xhigh-checkpoint8-undo-sync`;
- model: `Qwen/Qwen3.8-27B-FP8` at immutable Hugging Face revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- reasoning: `xhigh`, supplied as a vLLM default chat-template argument;
- source change: execute at most eight actions from one model-requested batch,
  then return the settled board, limit, unexecuted suffix, and re-grounding
  instruction;
- adapter fixes: expose engine action 7 as `UNDO` in both directions; after an
  automatic reset, clear stale conversational/world-model state, persist the
  reset result, and stop without another model request if reset remains
  terminal;
- runtime: concurrency 28, analyzer timeout 900 seconds, per-game runtime cap
  7,920 seconds, one pass, plus a global soft deadline at 30,600 seconds from
  notebook start (30 minutes before the pinned nine-hour deployment limit);
- inference: server/analyzer contexts 65,536/32,768, temperature/top-p/top-k
  `0.6 / 0.95 / 20`, current-grid image transport at 4x.

The checkpoint does not discard the remaining actions silently. They are
returned to the same Python REPL, while the model is told to re-ground on the
settled board before deciding whether that suffix is still valid.

## Required Kaggle Inputs

The notebook expects these attached Kaggle inputs:

- `jeroencottaar/taaf-kaggle-source-share` (version 4)
- `driessmit1/arc3-vllm-h100-wheelhouse-v3` (version 1)
- `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (version 1)
- the competition dataset mounted at
  `/kaggle/input/competitions/arc-prize-2026-arc-agi-3`

Kaggle may mount datasets either at `/kaggle/input/<slug>` or at
`/kaggle/input/datasets/<owner>/<slug>`. The notebook records the resolved paths
in `TAAF_KAGGLE_INPUT_PATHS` for the setup commands and solver. If the model
dataset contains one nested snapshot directory, the launcher resolves that
directory and records the final path. Multiple candidate roots fail closed.

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
`/kaggle/working/taaf_run_summary.txt` with non-secret run metadata: candidate
and model identity, immutable model revision, source and post-patch hashes,
model-validation status, submission mode, Python/platform info, known safe
TAAF/Kaggle environment knobs, attached input refs and resolved paths, source
and writable bundle paths, selected solver attributes, the validated deployment
budget and global soft deadline, and offline selected game ids. Non-submission
runs may also render `diagnostics.html` inline.

The launcher supports a few environment knobs for iteration. `TAAF_OFFLINE_GAMES`
and `TAAF_OFFLINE_MAX_GAMES` only apply to non-submission Save & Run offline
games. `TAAF_BENCHMARK_CONCURRENCY`, `TAAF_ANALYZER_TIMEOUT`,
`TAAF_MAX_ACTIONS_PER_GAME`, and `TAAF_MAX_RUNTIME_S_PER_GAME` override the
matching `bm.solver` attributes after the candidate defaults are loaded.
Invalid values raise clear launcher errors. Overrides intentionally make a run
an experiment rather than an exact candidate reproduction, and the manifest
records them.

## Packaging and Output Audit

`scripts/build_kaggle_package.py` turns the source notebook into a reviewed
Kaggle package without contacting Kaggle. It requires an explicit
`<username>/<slug>` id, emits private/no-internet metadata for one RTX Pro 6000,
pins the competition, datasets, and Docker image, clears notebook execution
artifacts, compiles every generated code cell, and refuses to replace an output
directory it does not own. Preflight mode injects a one-off offline game limit;
full mode remains code-equivalent to the cleaned source notebook. Generated
packages live under ignored `build/` paths and are not source artifacts.

`scripts/audit_kaggle_output.py` is the fail-closed boundary after a user
downloads an exact finished Kaggle kernel version. It does not modify the output
directory. It verifies:

- terminal manifest milestone/outcome, candidate and immutable model identity,
  checkpoint limit, dataset versions, source/patched tree digests, wheelhouse
  SHA-256 status, and model CRC/inventory status;
- default solver concurrency/timeouts, unless explicitly accepted positive
  overrides match their recorded `TAAF_*` environment values;
- one benchmark pass, selected/run game counts, public-game ids for offline
  modes, terminal non-crashed game states, finite scores, and TAAF's
  action-history invariant;
- a framed, non-empty `submission.parquet`, a non-empty vLLM server log without
  obvious traceback/OOM/fatal-startup markers, and mode-correct
  `diagnostics.html` presence.

The three audit modes are `preflight`, `full-offline`, and `submission`.
Submission count is taken from the live-gateway manifest instead of hard-coding
25 hidden games. To preserve TAAF's hidden-run diagnostics boundary, submission
mode requires full `benchmark.json` and `diagnostics.html` to remain absent and
accepts only the exact aggregate schema in `taaf_submission_health.json`: run
counts and clean/problem totals, with no ids, scores, actions, frames, or traces.
Passing the auditor proves artifact consistency and run completion under these
contracts; it does not prove a leaderboard score or a causal improvement.

Kaggle authentication, `kaggle kernels push`, waiting, exact-version output
download, and any competition rerun are user-run external operations. The local
builder and auditor never authenticate, upload, launch, submit, or mutate Kaggle
state.

## Evidence Boundary

The candidate is selected from public external measurements in
`sonpham-org/arc-3` at commit
`3ba35ce5a9a81efc068f0b40db1b1e92c785d588`. Its public 25-game artifacts
report checkpoint-8 Qwen3.8 runs of `3.8361`, `5.8142`, and `4.4358`; the first
two average `4.8251`. Published no-checkpoint replicas using the Kaggle 4x
visual shape report `3.5063` and `5.3877`, averaging `4.4470`. Because the first
cap-8 pair used the native/8x visual setup, these local groups are supporting
evidence rather than a clean one-variable estimate of checkpoint-8. The same
public evidence records Kaggle submission `55551321` at `2.03`.

The previous repository candidate completed an audited 25-public-game Kaggle
run on 2026-09-08 with mean score `4.671335090995715`, 27 total levels cleared,
and positive score on 18 of 25 games. Its traces contained 27 `ACTION7` attempts,
seven explicit unknown-action errors, 62 stale game-over prompt occurrences,
and 452 analyzer turn-budget yields. Those counts motivated the two adapter
fixes, but they do not establish a score gain. Local checks can prove patch
fidelity and syntax, but cannot serve or exercise the 31 GB model on this
machine's RTX 4050. A new RTX Pro 6000 preflight/full run and competition rerun
remain required before making a score claim for the `undo-sync` candidate.

## Deliberately Deferred Follow-ups

- HUD/no-impact filtering: public external evidence suggests it may help some
  games, but it is materially more invasive. It is deferred until the minimal
  Qwen3.8/checkpoint-8 candidate is reproduced.
- Broader prompt, state-memory, graph, or animation changes are excluded from
  the current candidate because they confound the measured model/checkpoint
  effect and have not earned inclusion here.

## Preserved Runtime Assets

The broad official/runtime assets remain in this repo for now:

- `ARC-AGI-3-Agents/`
- `arc_agi_3_wheels/`
- `environment_files/`
- `.gitignore`, `LICENSE`, and `Dockerfile`

These are kept because they are official/runtime support files or useful local
reference material, and they are not part of the removed CWM/RL baseline.

## Risks and Caveats

- GPU and compatibility dependency: the notebook depends on the attached TAAF
  bundle, 31 GB FP8 Qwen snapshot, vLLM 0.19.0 wheelhouse, and one RTX Pro 6000.
  Metadata and shard validation does not prove that this public Kaggle model
  copy will load with the wheelhouse; only the setup smoke test can do that.
- Timeout risk: setup starts local analyzer infrastructure and then runs the
  benchmark. The pinned deployment payload grants 32,400 seconds and the
  launcher now reserves the final 1,800 seconds for scorecard closure and
  teardown. Hitting that guard produces a partial/cancelled run that the strict
  auditor rejects as a clean reproduction, but avoids relying on a hard kill.
- Source coupling: exact hashes deliberately reject upstream TAAF or model
  changes. Updating either attachment requires a conscious re-audit and new
  expected hashes.
- Pickle boundary: both benchmark pickle files are pinned by exact hash before
  loading, but Python pickle is still executable code. Only the named public
  Tufa dataset belongs in this boundary.
- Attribution and baseline caveat: this path uses the public Tufa Labs duck
  harness/milestone submission approach. Any use should preserve attribution and
  account for competition rules and public-baseline expectations.
- Reproducibility: this repository alone is not a complete submission artifact.
  The exact Kaggle datasets, RTX Pro 6000 image, competition environment, and
  generated manifest are part of the run provenance.
