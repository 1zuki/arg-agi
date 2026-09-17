# Flash-Next/MTP no-overlay c12 — Kaggle handoff

## Which file?

**Full candidate notebook for Save & Run, then competition submission:**

[`full/arc-agi3-qwen38-flash-next-mtp-full.ipynb`](full/arc-agi3-qwen38-flash-next-mtp-full.ipynb)

**Corrected short preflight (test first; do not submit this package):**

[`preflight/arc-agi3-qwen38-flash-next-mtp-preflight.ipynb`](preflight/arc-agi3-qwen38-flash-next-mtp-preflight.ipynb)

These notebooks contain their launcher and teardown patch; no local Python files
need to be uploaded alongside the notebook in the Kaggle web editor. They still
require the external Kaggle inputs below. `kernel-metadata.json` is for CLI use,
not a competition prediction file. Its owner is a placeholder: regenerate the
package with your own authorized Kaggle account before pushing.

## Status and risks

This is an **unverified experiment**, not the notebook that scored 3.40.
Keep submission `56170646` (`izukia/arc-agi3-flash-next-mtp-full/1`, public
**3.40**) as the incumbent. The root `arc-agi-flash-next-mtp.ipynb` is the baseline
source, not this generated c12 handoff. The older state-overlay c12 submission
scored 1.80 and is not this candidate.

The corrected preflight and full candidate have **not run on Kaggle**: our last
upload was rejected by the weekly GPU quota. An earlier 25-game c12 preflight
finalized 12 games and cancelled 13 because its global deadline allowed only one
worker batch. It is rejected. The preflight here fixes that deadline to allow
three batches. The clean 25-game c28 short control scored 1.4458934228164997
**offline**, with 648 actions; it is not a leaderboard score. Earlier 12-game
c28/c12 runs did not isolate concurrency because both could run all 12 at once.
No result currently proves c12 is better than the incumbent.

| Setting | Preflight | Full candidate |
| --- | --- | --- |
| Duck game concurrency | 12 | 12 |
| Public offline games / passes | 25 / 1 | 25 / 1 |
| Per-game runtime | 1,800 s | 7,920 s |
| Analyzer timeout | 900 s | 900 s |
| Offline deadline | 5,400 s after setup + 120 s grace | Notebook start + 31,800 s |
| Reasoning-state overlay | **Disabled** | **Disabled** |

The full run needs roughly 6.6 hours of gameplay at the per-game cap (three worker
batches), plus setup. Hidden competition game counts can differ; the competition
rerun retains the nine-hour target and leaves its global deadline to Kaggle's
gateway. Lower concurrency may hurt hidden-game coverage. Prefix caching remains
disabled; vLLM retains eight sequences, MTP three tokens, and the pinned serving
profile. The audit verifies serving identity and recorded artifacts, not every
model/runtime payload hash. Do not turn on `--agent-state-patch` for this candidate.

## Prerequisites / attached inputs

Use the competition's permitted team/code-sharing workflow. Every runner must
have authorized access, accept the competition rules, and have permitted GPU
resources. Do not share account credentials or use extra accounts to evade quotas
or submission limits. Confirm team sharing/submission rules before a friend runs it.

- Competition: `arc-prize-2026-arc-agi-3`
- Dataset: `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`
- Dataset: `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`
- Model: `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`
- Accelerator: **RTX Pro 6000** (`NvidiaRtxPro6000`), not a T4/P100 substitute
- Internet: **off**; keep the notebook private while experimenting
- Pinned Docker image and remaining input metadata: see the package metadata files

For the web UI: import the relevant `.ipynb`, attach all four inputs, select the
correct GPU/image/settings, and use **Save Version → Save & Run All**. Running
against offline environment files is only validation; it is not a submission.
For a reproducible CLI setup, follow the steps below (requires Python 3.10+ and `uv`).

## Recommended CLI workflow

Clone the repo, then set your authorized Kaggle username:

```bash
git clone https://github.com/1zuki/arg-agi.git
cd arg-agi
export KAGGLE_OWNER='your-real-kaggle-username'
uvx --from kaggle kaggle auth login
```

### 1. Run corrected preflight

Regenerate metadata for your account (does not contact Kaggle):

```bash
python scripts/build_flash_next_package.py \
  --kernel-id "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-preflight" \
  --mode preflight --max-games 25 --concurrency 12 \
  --runtime-seconds 1800 --terminal-grace-seconds 120 \
  --analyzer-timeout 900 --output-dir build/friend-flash-c12-preflight

uvx --from kaggle kaggle kernels push -p build/friend-flash-c12-preflight -t 30000
```

Record the **actual accepted version** from the push response. Check until it is
`COMPLETE`, then download to a **fresh** directory and audit:

```bash
uvx --from kaggle kaggle kernels status \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-preflight"

uvx --from kaggle kaggle kernels output \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-preflight" \
  -p build/friend-flash-c12-preflight-output

uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode preflight --expected-games 25 --expected-concurrency 12 \
  --expected-runtime-seconds 1800 --expected-gameplay-budget-seconds 5400 \
  --expected-terminal-grace-seconds 120 --expected-analyzer-timeout 900 \
  --require-clean build/friend-flash-c12-preflight-output
```

Do not publish another version under that slug until its outputs are downloaded.
The inspected Kaggle CLI parses a `/version` suffix for status/output but does
not send that version to these endpoints: they return the latest session.
Separate slugs for preflight/full and one run at a time prevent mixing artifacts.
Compare the clean short-preflight score/throughput with the c28 control above;
a single noisy offline comparison does not prove a public score improvement.

### 2. Run the full candidate only if preflight is clean and promising

```bash
python scripts/build_flash_next_package.py \
  --kernel-id "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-full" \
  --mode full --concurrency 12 --output-dir build/friend-flash-c12-full

uvx --from kaggle kaggle kernels push -p build/friend-flash-c12-full -t 30000
```

Record the new actual version; wait for `COMPLETE`, download, and require the full
strict audit to pass:

```bash
uvx --from kaggle kaggle kernels status \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-full"

uvx --from kaggle kaggle kernels output \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-full" \
  -p build/friend-flash-c12-full-output

uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode full-offline --expected-games 25 --expected-concurrency 12 \
  --expected-runtime-seconds 7920 --expected-analyzer-timeout 900 \
  --require-clean build/friend-flash-c12-full-output
```

Reject crashes, cancelled/unfinished games, configuration/digest mismatch, or
failed teardown. Review the score and per-game regressions before submission;
the incumbent full offline mean was 5.502654474314745. Do not combine outputs
from different runs to fabricate a clean bundle.

### 3. Submit the notebook version, not the offline placeholder

Only after the **full** audit and score review pass, set the version from the
full push response (do not assume it is 1):

```bash
export FULL_VERSION='actual-full-version-number'
uvx --from kaggle kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-nostate-c12-full" \
  -v "$FULL_VERSION" -f submission.parquet \
  -m "Flash-Next MTP no-overlay c12; audited public25"

uvx --from kaggle kaggle competitions submissions arc-prize-2026-arc-agi-3
```

This submits a Kaggle **notebook rerun**. Save & Run produces a dummy offline
`submission.parquet`; **never upload that local file as predictions**. In the
web UI, submit the completed **full notebook version**, not the preflight.
Promote only when a new public score exceeds 3.40; offline means are not public scores.

## Local checks and attribution

`uvx --with pytest pytest -q tests` and `git diff --check`. Tests that require
locally downloaded source fixtures may skip on a fresh clone; no GPU gameplay
or public score is established by these tests. The published notebook code is
identical to the generated local packages; only markdown notes and metadata
owner/title templates were adjusted for sharing.

Full credit to Tufa Labs/Jeroen Cottaar for the Duck harness and to Keith Tyser
for the public Flash-Next/MTP serving stack. Keep their attribution and respect
their sources' licenses and competition rules when sharing or submitting.
