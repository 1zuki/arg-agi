# Flash-Next/MTP ACTION7 passthrough — experimental c28 handoff

**Completed experiment; public submission `56369560` scored 2.32. Keep submission `56170646` at public 3.40.**

This candidate adds one missing mapping to the Flash adapter:

```python
"ACTION7": "ACTION7",
```

The baseline advertises `ACTION7` but rejects that name on input. The change
allows it through the parser; the existing engine `available_actions` gate still
rejects it when unavailable. No forced action, game-specific policy, or assumption
that action 7 means undo is introduced.

## Evidence and scope

The saved 3.40 baseline has **33 Python tool snippets mentioning ACTION7 across
six games, and zero ACTION7 executions**. Snippet mentions are not exact attempted
action counts: a branch may not execute, and a loop may execute more than once.

Reproduce the diagnostic without modifying the saved run:

```bash
python scripts/diagnose_flash_action7.py build/kaggle-output/flash-full-v1
```

The downloaded pinned solver's real normalization/dispatch methods were also
exercised with an engine stub: baseline rejects the advertised action, candidate
executes it once, and unavailable ACTION7 remains rejected. This is an adapter
regression test, **not a model/gameplay score**.

An older *non-Flash* checkpoint-8 candidate bundled an UNDO alias with reset
synchronization. This is not that experiment: it retains the Flash 3.40 policy,
the neutral ACTION7 name, and no reset/state changes. No isolated Flash c28
ACTION7-only result appears in `FLASH_EXPERIMENTS_SINCE_3_40.md`.

Unchanged: Duck c28, 900-second analyzer timeout, 7,920-second full per-game cap,
model/runtime, eight vLLM sequences, MTP three tokens, prefix caching disabled,
prompts, sampling, scorer, game order, and teardown. The failed reasoning-state
overlay is disabled and cannot be combined with `--action7-patch`.

| Identity | SHA-256 |
|---|---|
| Original action_names.py | `d359c22dd9f3925ed54cb05a60640e8aa1121c226a068b4736e8dfad46d2d115` |
| Patched action_names.py | `07abd197442bc1d075067717b7dd357f39f9385d90e1ecdeebe6d5de199ae156` |

Patch name: `action7-passthrough-v1`. Only the small mapping file is overlaid;
source drift, an already-imported source module, or a wrong imported path/hash
aborts rather than silently running an unpatched candidate. Tufa Labs/Duck and
Keith Tyser's Flash serving attribution remain in the notebook.

## Completed remote checkpoints (verified 2026-10-26)

- Kernel: `izukia/arc-agi3-flash-next-mtp-action7-preflight25/1`
- Status: `COMPLETE`.
- Strict audit: 25 games, c28, 1,800-second per-game budget, 592 actions,
  clean teardown, offline mean **2.0970268186983443**.
- Artifact: `build/kaggle-output/flash-action7-preflight25-20260919-v1/`.
- Full kernel: `izukia/arc-agi3-flash-next-mtp-action7-full/1`, also `COMPLETE`.
- Full strict audit: 25 games, c28, 7,920-second per-game budget, 3,791
  actions, clean teardown, offline mean **5.975441389862917**.
- Full artifact: `/tmp/arc3-action7-full-output-20261026/`.
- Public submission `56369560` scored **2.32**, below the protected **3.40**;
  neither the offline mean nor the completed artifact is a promotion signal.
- Preflight notebook SHA-256: `649001bd663ce954e211bc6347a87917c4fb3eff2d5ced280f1649f314da2003`
- Full notebook SHA-256: `116b4da2078316820569fc48f27ed3a2dfbef4691688c2479e38297570b7cf56` (prepared only).
- The status and download gates above are complete. The build/push commands below
  are for reproduction only; do not push another run without a new hypothesis.
- Set `KAGGLE_OWNER=izukia` and `PREFLIGHT_VERSION=1` for this run.

## Build / run / audit

Already-generated local packages (not downloaded remote results):

- `build/flash-action7-preflight25-20260919/`
- `build/flash-action7-full-20260919/`

Both target `izukia`; rebuild with your authenticated owner if different. Use fresh
output paths, and retain the exact version number returned by each push. Required
GPU, attachments, and offline settings are in the generated metadata.

```bash
KAGGLE_OWNER=izukia
python scripts/build_flash_next_package.py \
  --kernel-id "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-preflight25" \
  --mode preflight --max-games 25 --concurrency 28 --runtime-seconds 1800 \
  --terminal-grace-seconds 120 --analyzer-timeout 900 --action7-patch \
  --output-dir build/flash-action7-preflight25-20260919

python scripts/build_flash_next_package.py \
  --kernel-id "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-full" \
  --mode full --concurrency 28 --action7-patch \
  --output-dir build/flash-action7-full-20260919
```

### 1. Clean preflight first

```bash
uvx --from kaggle kaggle kernels push -p build/flash-action7-preflight25-20260919
# Set to the version actually returned by push; never assume latest == reviewed.
PREFLIGHT_VERSION='<version>'
uvx --from kaggle kaggle kernels status \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-preflight25"
# Wait for COMPLETE, then:
uvx --from kaggle kaggle kernels output \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-preflight25/$PREFLIGHT_VERSION" \
  -p build/kaggle-output/flash-action7-preflight25-20260919
uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode preflight --expected-games 25 --expected-concurrency 28 \
  --expected-runtime-seconds 1800 --expected-gameplay-budget-seconds 1800 \
  --expected-terminal-grace-seconds 120 --expected-analyzer-timeout 900 \
  --kernel-log arc-agi3-flash-next-mtp-action7-preflight25.log \
  --require-action7-patch --require-clean \
  build/kaggle-output/flash-action7-preflight25-20260919
```

There is one worker batch. Require the declared budget/grace; do not change audit
expectations merely to accept a mismatched run. Inspect the diagnostic on the new
output too. If no ACTION7 is exercised, that does not falsify the mapping fix, but
it gives no gameplay evidence for its proposed benefit.

### 2. Full gate only after clean preflight

```bash
uvx --from kaggle kaggle kernels push -p build/flash-action7-full-20260919
FULL_VERSION='<version>'
uvx --from kaggle kaggle kernels status \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-full"
# Wait for COMPLETE, then:
uvx --from kaggle kaggle kernels output \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-action7-full/$FULL_VERSION" \
  -p build/kaggle-output/flash-action7-full-20260919
uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode full-offline --expected-games 25 --expected-concurrency 28 \
  --expected-runtime-seconds 7920 --expected-analyzer-timeout 900 \
  --require-action7-patch --require-clean \
  build/kaggle-output/flash-action7-full-20260919
```

Review all 25 games, levels cleared, action efficiency, errors, and runtime, not
only the six motivating games. Added actions can waste moves or lead to worse
plans; mapping correctness does not imply a score gain. Sampling remains the
baseline's stochastic policy, not a seeded paired experiment. A contemporary
c28 control/repeat is preferable if GPU quota allows; old baseline differences
alone do not establish causality.

Only consider a public rerun from the **exact completed/audited full version**.
Never submit the local offline placeholder as predictions. Replace 3.40 only with
a strictly higher public score. The private 25-game preflight and full run were
uploaded as version 1 on 2026-09-19, completed, and were audited on 2026-10-26.
The full run's public submission scored 2.32, so the 3.40 incumbent remains
protected.

## Local verification

```bash
# Optional pinned solver fixture for the real-adapter/stub-engine dispatch test:
uvx --from kaggle kaggle datasets download \
  -d keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 \
  -f src/ARC3-Inference/inference/framework/solver.py \
  -p build/flash-action7-source-20260919 --quiet
python -m unittest discover -s tests -v
```

The dispatch test checks solver SHA-256
`65ddfcdbe8c24fdc31246ff018b51469827b21c38995243a36a818adbed33370`;
it explicitly skips if absent. Other new mapping/import/audit tests use the small
committed fixture and do not need model weights. The RTX 4050 here cannot serve
the pinned model. Full model/runtime payload hashing remains a known limitation
of the existing Flash fast-setup contract.
