# Flash-Next/MTP c8-r2 handoff

This is the best measured offline improvement candidate currently available. It
keeps the pinned Flash-Next/MTP model and serving profile, changes Duck game
concurrency to **8**, and caps each game at **7,200 seconds** so all four public
worker batches fit inside Kaggle's nine-hour notebook budget.

Evidence boundary:

- Protected incumbent: submission `56170646`, public score **3.40**.
- Earlier c8 artifact: offline mean **9.05124079307816**, but rejected because
  the last game was cancelled at the nine-hour boundary.
- c8-r2 has not run on Kaggle yet. It must pass the strict full audit before
  submission.
- The current Kaggle account is blocked by the weekly GPU quota, so no run was
  launched from this account.

## Build the package

Use an authorized Kaggle account and do not use extra accounts to evade quotas
or submission limits:

```bash
git clone https://github.com/1zuki/arg-agi.git
cd arg-agi
export KAGGLE_OWNER='your-real-kaggle-username'

python scripts/build_flash_next_package.py \
  --kernel-id "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-c8-full-r2" \
  --mode full --concurrency 8 --full-runtime-seconds 7200 \
  --title 'arc agi3 flash next mtp c8 full r2' \
  --output-dir build/flash-c8-full-r2
```

Attach the inputs from `kaggle/flash-next-nostate-c12/README.md`, use an RTX
Pro 6000, and keep internet disabled. Push the private package:

```bash
uvx --from kaggle kaggle kernels push -p build/flash-c8-full-r2 -t 30000
```

Wait for the exact kernel to report `COMPLETE`, then download its output into a
fresh directory and audit it:

```bash
uvx --from kaggle kaggle kernels status \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-c8-full-r2"

uvx --from kaggle kaggle kernels output \
  "$KAGGLE_OWNER/arc-agi3-flash-next-mtp-c8-full-r2" \
  -p build/kaggle-output/flash-c8-full-r2

uvx --with pyarrow --from kaggle python scripts/audit_flash_output.py \
  --mode full-offline --expected-games 25 --expected-concurrency 8 \
  --expected-runtime-seconds 7200 --expected-analyzer-timeout 900 \
  --require-clean build/kaggle-output/flash-c8-full-r2
```

Only if that audit passes and the offline mean is reviewed, rerun the completed
full notebook as the competition submission. Submit that completed notebook
version, never the local offline placeholder `submission.parquet`, and retain
the 3.40 incumbent unless the new public score is higher.
