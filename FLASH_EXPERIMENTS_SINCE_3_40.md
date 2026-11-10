# ARC-AGI-3 Flash-Next/MTP Experiment Ledger Since the 3.40 Baseline

**Repository:** `1zuki/arg-agi`
**Scope:** experiments and engineering work attempted after the protected Flash-Next/MTP 3.40 submission
**Report date:** 2026-09-24
**Competition:** `arc-prize-2026-arc-agi-3`

> **Bottom line:** the protected Flash-Next/MTP baseline remains the best public result.
> Submission `56170646` scored **3.40**. The later completed public submissions recorded
> in the current reconciliation scored **1.80**, **2.54**, **2.81**, **2.45**, **2.32**,
> and **2.80**; submission `56514817` remains pending. The large offline improvements
> observed in some variants did not transfer to the public leaderboard.

---

## 1. How to read this report

There are three different kinds of evidence in the experiment history:

1. **Offline benchmark evidence** — the mean from the saved `benchmark.json`/`score.json`
   artifacts over the fixed public game set. This is useful for debugging and comparing
   runtime behavior, but it is **not** a Kaggle leaderboard score.
2. **Strict artifact-audit evidence** — checks that the downloaded output contains the
   expected games, terminal states, serving identity, teardown artifacts, watchdog status,
   logs, scores, and configuration. A kernel completing is not the same as passing this
   gate.
3. **Public leaderboard evidence** — the score returned by Kaggle after the completed
   notebook was submitted and evaluated by the competition gateway. This is the promotion
   metric.

The report deliberately does not combine runs, fill in cancelled games, or convert an
offline mean into a claimed public score.

### Full-run acceptance rule used throughout

A candidate was only eligible for promotion if it had:

- all 25 pinned offline public games in the expected order;
- no cancelled or unfinished games in a full audit;
- the expected concurrency/runtime/analyzer configuration;
- valid score and benchmark coverage;
- clean vLLM/GPU teardown and preserved required artifacts;
- zero or bounded watchdog restarts, with the result recorded;
- a public leaderboard score above **3.40** before replacing the incumbent.

A preflight was allowed to cancel by deadline because it was a smoke test. A full run was
not.

---

## 2. Historical public scoreboard as of 2026-09-19

| Submission | Variant | Public score | Delta vs 3.40 | Decision |
|---:|---|---:|---:|---|
| `56170646` | Flash-Next/MTP baseline, c28 | **3.40** | 0.00 | **Protected incumbent** |
| `56249553` | Flash-Next/MTP state-overlay c12 | 1.80 | -1.60 | Rejected |
| `56265695` | Wuliao/Tufa public-source reproduction | 2.54 | -0.86 | Rejected |
| `56310095` | Flash-Next/MTP no-overlay c12 | 2.81 | -0.59 | Rejected |
| `56338677` | Flash-Next/MTP c8-r2 | 2.45 | -0.95 | Rejected |

The live Kaggle submission list marked all five entries `COMPLETE`. The UI screenshot
showed the c8-r2 notebook as **“arc agi3 flash next mtp c8 full r2 – Version 2”** with
public score **2.45**; the live submission listing identifies that result as `56338677`.

The public score is the only number used to decide whether to replace the baseline. The
offline means below are retained as diagnostic measurements, not leaderboard claims.

The current live reconciliation, including later scores and pending submissions, is in
Section 12.

---

## 3. Protected 3.40 baseline

### Identity

- **Submission:** `56170646`
- **Kernel:** `izukia/arc-agi3-flash-next-mtp-full/1`
- **Source notebook:** `arc-agi-flash-next-mtp.ipynb`
- **Runtime profile:** Flash-Next/MTP, Duck concurrency **28**
- **Per-game cap:** **7,920 seconds**
- **Pinned serving profile:** vLLM eight-sequence limit, MTP three speculative tokens,
  prefix caching disabled, pinned NVFP4 model/runtime and RTX Pro 6000 environment

### Validation evidence

The downloaded artifact is stored at:

```text
build/kaggle-output/flash-full-v1/
```

The strict offline audit passed:

- **25/25** games present, in the pinned public order;
- **4,683** actions;
- offline mean: **5.502654474314745**;
- every run reached a valid terminal state (`gave_up` in the saved benchmark;
  this is a harness terminal state, not a zero-quality claim);
- clean teardown, no surviving owned GPU process, and `shutdown_ok=true`;
- zero watchdog restarts;
- serving identity and required output artifacts preserved.

### Public result and interpretation

The completed competition rerun produced public score **3.40**. The offline mean
**5.502654474314745** is not comparable as if it were a public score: it comes from the
local/offline evaluation path and the public result comes from the Kaggle gateway's
competition execution/scoring path.

This run is the reference against which all later candidates were compared. It was not
replaced because no later public result exceeded 3.40.

---

## 4. Chronological experiment ledger

The entries below start after the 3.40 baseline and include both runs that reached Kaggle
and runs that were stopped by the validation gates before submission.

### 4.1 Full v2 teardown-reliability rerun

**Approximate date:** 2026-09-12
**Artifact:** `build/kaggle-output/flash-full-v2/`
**Hypothesis:** rerun the baseline package and check whether a slightly different execution
would improve the offline mean while preserving the same c28/full-runtime profile.

#### Configuration

- Flash-Next/MTP baseline behavior retained;
- concurrency **28**;
- 25 offline games;
- 7,920-second per-game cap;
- same pinned serving profile.

#### Observed result

- offline mean: **5.6051924776763435**;
- actions: **3,277**;
- all 25 benchmark entries were present and had valid benchmark terminal states;
- watchdog restarts: **0**;
- teardown: **failed** — `shutdown_ok=false`, with one stale owned GPU survivor.

The failure was not a normal gameplay score failure. The vLLM teardown gate retained a
GPU process row after shutdown, so the artifact was not considered clean. The run also
lacked the complete persisted teardown patch artifacts that were present in the accepted
baseline bundle.

#### Decision

**Rejected.** The offline mean was marginally higher than the baseline, but artifact
validity takes priority. A result with failed cleanup cannot be promoted merely because
its score is numerically higher.

#### Lesson

The audit must treat process/GPU teardown as part of correctness. “25 games completed”
and “higher offline mean” are insufficient if the runtime leaves owned resources behind.

---

### 4.2 c8 preflight

**Date:** 2026-09-12
**Kernel/artifact:** `izukia/arc-agi3-flash-next-mtp-preflight-c8/1`, downloaded as
`build/kaggle-output/flash-preflight-c8-v1/`
**Hypothesis:** reduce Duck concurrency from 28 to **8** to reduce contention against the
vLLM profile's effective eight-sequence limit and improve reasoning/action quality.

#### Configuration

- concurrency **8**;
- **8** games, not the full 25;
- 1,800-second preflight runtime;
- analyzer timeout 900 seconds;
- clean GPU-release teardown patch enabled.

#### Observed result

- offline mean: **5.768026065645113**;
- actions: **313**;
- all eight games were deadline-cancelled;
- teardown passed and `shutdown_ok=true`;
- watchdog restarts: **0**.

Cancellation was acceptable here because this was explicitly a bounded preflight. It was
not evidence that a full c8 candidate would complete.

#### Decision

**Passed as a preflight only.** The run justified testing a full c8 package, but it was
not eligible for competition submission.

#### Lesson

Preflight cancellation is a diagnostic outcome, not a full-run success. The full audit
must use stricter terminal-state rules.

---

### 4.3 Original c8 full run

**Date:** 2026-09-13 to 2026-09-14
**Artifact:** `build/kaggle-output/flash-c8-full-v1-20260914/`
**Hypothesis:** lower concurrency to c8 while keeping the original 7,920-second game cap;
allow each active game more effective model/service capacity and obtain a higher score.

#### Configuration

- concurrency **8**;
- 25 games;
- 7,920 seconds per game;
- four worker batches are required because `ceil(25 / 8) = 4`;
- nine-hour Kaggle notebook target: 32,400 seconds.

#### Observed result

- offline mean: **9.05124079307816**;
- actions: **17,386**;
- 24 games reached valid terminal states;
- final game `tr87-cd924810` was **cancelled** near the global nine-hour deadline;
- teardown passed and `shutdown_ok=true`;
- watchdog restarts: **0**.

The number looked very promising offline, but the schedule itself was impossible to
complete safely: four batches at 7,920 seconds require up to 31,680 seconds of gameplay
before allowing meaningful setup/teardown reserve. The final game was cancelled at the
notebook boundary.

#### Decision

**Rejected as an invalid full run.** The mean is preserved in the ledger for historical
comparison, but it is not a valid 25-game candidate and was never promoted or submitted.

#### Root cause

The c8 change improved per-game runtime allocation but did not account for the number of
worker batches. Lowering concurrency increases the number of serial batches. A valid
schedule must satisfy:

```text
ceil(number_of_games / concurrency) * per_game_runtime + reserve <= 32,400 seconds
```

This run violated the practical form of that constraint.

---

### 4.4 Targeted `tr87-cd924810` cancellation diagnosis

The cancelled last game was isolated rather than silently spliced into the failed c8
bundle. Two targeted preflights were run.

#### Attempt A: original deadline behavior

- artifact: `build/kaggle-output/flash-c8-tr87-preflight-v1-20260914/`;
- c8 configuration;
- one game: `tr87-cd924810`;
- configured runtime: 7,920 seconds;
- final state: `cancelled`;
- final wallclock: about **7,294 seconds**;
- offline game score: **26.98336045995675**;
- actions: **1,516**;
- teardown: clean.

This run showed that the game could make substantial progress but still be cancelled by
the bounded notebook deadline. It did not prove that the game would finish if inserted
into the original full bundle.

#### Attempt B: deadline measured after serving setup

- artifact: `build/kaggle-output/flash-c8-tr87-preflight-v2-20260915/`;
- one game: `tr87-cd924810`;
- post-setup gameplay deadline instrumentation;
- final state: still `cancelled` at the runtime boundary;
- offline game score: **9.3679593590361**;
- actions: **1,409**;
- teardown: clean.

The second attempt reproduced the cancellation and showed that this was not simply an
unclean vLLM shutdown. The targeted run was useful for diagnosing deadline accounting,
but its score was noisy and it was not valid evidence for replacing a 25-game result.

#### Decision

**Diagnostic only; no promotion.** Results from either targeted run were not merged into
a full candidate.

#### Engineering outcome

The builder and auditor were extended to support:

- explicit pinned game selection;
- configurable runtime and concurrency;
- post-setup deadline instrumentation;
- terminal grace periods;
- stricter expected-game and expected-configuration checks.

These changes made it possible to distinguish a model/runtime failure from a schedule
failure instead of masking the issue by combining artifacts.

---

### 4.5 State-overlay c12 preflight

**Date:** 2026-09-14
**Artifact:** `build/kaggle-output/flash-state-c12-preflight-v2/`
**Hypothesis:** combine lower concurrency (**12**) with a digest-bound reasoning/world-model
overlay that preserves labelled state emitted in the model's reasoning field between tool
calls. The expected benefit was better continuity across analyzer turns, not a new model.

#### Configuration

- concurrency **12**;
- 12-game preflight;
- 1,800-second per-game cap;
- post-setup gameplay deadline;
- reasoning/world-model overlay enabled;
- clean teardown patch enabled.

#### Observed result

- offline mean: **4.2940997831044125**;
- actions: **701**;
- 12/12 games reached valid terminal states;
- overlay patch identity and import/reassertion evidence were present;
- teardown passed;
- watchdog restarts: **0**.

#### Decision

**Passed as a candidate preflight.** The overlay was technically integrated and the
preflight was clean enough to justify the full run. This did not establish a public-score
improvement.

---

### 4.6 State-overlay c12 full run and public submission

**Date:** 2026-09-14 to 2026-09-15
**Kernel:** `izukia/arc-agi3-flash-next-mtp-state-c12-full/1`
**Artifact:** `build/kaggle-output/flash-state-c12-full-v1/`
**Public submission:** `56249553`
**Hypothesis:** the c12 service-capacity change plus persistent reasoning state would improve
long-horizon game solving over c28.

#### Configuration and implementation

The overlay was implemented in `scripts/flash_agent_state_patch.py` as a digest-bound
patch:

- it refused to patch if the pinned tool-agent source digest changed;
- it broadened parsing of labelled state lines such as `KEY: value`;
- it fed labelled state emitted in the reasoning field into the agent's summarized
  knowledge before the next analyzer turn;
- the package recorded the source and patched-file digests in the output;
- the strict auditor required the overlay identity and import evidence for this variant.

The full run used:

- concurrency **12**;
- all 25 public games;
- 7,920 seconds per game;
- the same pinned Flash-Next/MTP serving stack;
- clean GPU-release teardown.

#### Strict offline result

The strict full audit passed:

- **25/25** games;
- offline mean: **7.0094912357041395**;
- actions: **11,288**;
- all games reached valid full-run terminal states;
- overlay digest/import/reassertion checks passed;
- teardown `shutdown_ok=true`;
- watchdog restarts: **0**.

This was a valid audited offline candidate, and its offline mean exceeded the baseline's
offline mean. It still required the public competition gate.

#### Public result

The candidate was submitted as `56249553` and received public score **1.80**.

#### Decision

**Rejected.** This is the clearest example that a strong offline score and a strict local
artifact audit do not guarantee a stronger public score. The 3.40 baseline remained
protected.

#### Lesson

The state overlay changed behavior enough to hurt the hidden/public execution path. It
should not be re-enabled without a new mechanism and a fresh, independently justified
hypothesis.

---

### 4.7 Wuliao/Tufa exact public-source reproduction

**Date:** 2026-09-15
**Artifact:** `build/kaggle-output/wuliao-anim-repro-v1-20260915-2059/`
**Public submission:** `56265695`
**Hypothesis:** reproduce the public Apache-2.0 Wuliao/Tufa animation-oriented approach
rather than continuing to tune only Flash concurrency, and test whether a known public
implementation transferred better to the competition gateway.

#### Configuration and observed offline result

The saved local artifact reports:

- 25 games;
- c28 runtime profile;
- offline mean: **6.26606582050872**;
- actions: **2,975**;
- 25/25 games with valid saved terminal states;
- clean teardown with `shutdown_ok=true`;
- no saved watchdog restart evidence.

The Kaggle submission description identified it as an **Apache-2.0 Wuliao/Tufa exact
public-source reproduction** and an audited 25-game run. The public submission list marked
it complete.

#### Public result and decision

Public score: **2.54**.

**Rejected.** It underperformed the 3.40 Flash baseline despite a respectable offline
mean. The reproduction did not provide a public-score improvement.

#### Lesson

Switching to a different public strategy is a valid experiment, but public-source
reproduction alone is not evidence that the strategy fits this specific model/runtime,
competition gateway, or hidden game distribution.

---

### 4.8 Short c28/c12 controls and the batch-budget bug

**Date:** 2026-09-16 to 2026-09-17
**Artifacts:**

- `build/kaggle-output/flash-preflight-c28-control1200-v1/`
- `build/kaggle-output/flash-preflight-c12-candidate1200-v1/`
- `build/kaggle-output/flash-c28-short-control25-v1/`
- `build/kaggle-output/flash-c12-short-deadline25-v1/`

**Hypothesis:** compare c28 and c12 under controlled short runtimes before spending another
full Kaggle run, and determine whether lower concurrency produced more useful actions or
higher offline scores.

#### Earlier 12-game comparison

The 12-game exploratory runs completed cleanly:

| Control | Concurrency | Games | Offline mean | Actions | Teardown |
|---|---:|---:|---:|---:|---|
| c28 control | 28 | 12 | 2.3971360802492994 | 572 | clean |
| c12 candidate | 12 | 12 | 3.905293961267125 | 710 | clean |

This comparison was not sufficient to prove a concurrency win. Both settings could run
all 12 games in a single worker batch, so it did not exercise the multi-batch scheduling
problem that matters for the full 25-game workload.

#### 25-game short control

The clean c28 control completed all 25 games:

- concurrency **28**;
- 1,800-second cap;
- offline mean: **1.4458934228164997**;
- actions: **648**;
- 25/25 valid terminal states;
- clean teardown.

The c12 short candidate used an incorrect one-batch global deadline and produced:

- concurrency **12**;
- offline mean: **1.3983318512083465**;
- actions: **719**;
- 12 games validly finalized;
- **13 games cancelled**;
- clean teardown.

The c12 number was therefore not a valid apples-to-apples 25-game result. It was retained
as a debugging artifact, not as evidence that c12 was better or worse.

#### Engineering fix

The package builder was corrected to calculate the number of batches as:

```python
ceil(number_of_games / concurrency)
```

and to budget the short run accordingly. The same reasoning was later embedded in the
full-package schedule guard. This prevented a package from silently claiming a 25-game
experiment while its global deadline only allowed one c12 batch.

#### Decision

**Diagnostic/control only.** No public submission was made from these short controls.

---

### 4.9 No-overlay c12 handoff and direct public submission

**Engineering handoff commit:** `c8deb63` (`feat: publish experimental Flash no-overlay c12 Kaggle handoff`)
**Public submission:** `56310095`
**Public score:** **2.81**
**Hypothesis:** remove the reasoning-state overlay, keep c12's lower concurrency, and test
whether the poor c12 public result was caused by the overlay rather than concurrency.

#### Intended package

The handoff package was explicitly marked experimental and identified the no-overlay
full notebook as the file to run. Its intended configuration was:

- concurrency **12**;
- no reasoning-state overlay;
- Flash-Next/MTP serving profile otherwise retained;
- preflight first, then a strict full 25-game audit;
- submit only the completed notebook version, never the local offline placeholder
  `submission.parquet`.

The README correctly preserved the 3.40 incumbent and documented the earlier c12 deadline
mistake.

#### What actually happened

The no-overlay c12 candidate was submitted directly and received public score **2.81**.
The required clean preflight/full-audit chain was not completed before that submission.
The exact remote kernel artifact for this submission is not available through the current
owner account, so this ledger intentionally does **not** claim a strict audit result for
`56310095`.

#### Decision

**Rejected.** Public score 2.81 is below 3.40. The experiment also exposed a process
failure: a package README can document the correct gate, but the gate still has to be
followed operationally before submission.

#### Lesson

The no-overlay result was useful diagnostically: it did not recover the baseline public
score. Future candidates need a recorded full audit before any submission decision.

---

### 4.10 c8-r2 quota-safe/schedule-safe rerun

**Engineering commit:** `f1b15a9` (`feat: prepare quota-safe Flash c8 rerun`)
**Local package:** `build/flash-c8-full-r2-20260918/`
**Public result:** submission `56338677`, score **2.45**
**Hypothesis:** retain c8's potential service-capacity benefit but reduce the per-game cap
so all four public batches fit inside the fixed nine-hour Kaggle budget.

#### Package change

The builder gained a full-runtime schedule guard. The c8-r2 contract was:

- concurrency: **8**;
- per-game runtime: **7,200 seconds**;
- public batches: `ceil(25 / 8) = 4`;
- reserve: **3,600 seconds**;
- total: `4 * 7,200 + 3,600 = 32,400 seconds`.

The builder rejects a full package when the requested runtime and concurrency cannot fit
inside the fixed 32,400-second Kaggle budget after reserve. Local preparation checks
reported **63 passed, 1 skipped, and 15 subtests passed**.

The package metadata targeted the pinned model/runtime, GPU, Docker image, competition,
and offline settings. No new model training was introduced; this was a runtime/schedule
experiment.

#### Remote evidence boundary

The original owner account was blocked by its weekly GPU quota during preparation. The
current credentials cannot retrieve the exact c8-r2 kernel slug/artifact (`404`), so the
strict downloaded-artifact audit status for this public submission is unknown in this
report. The public UI and live submission listing do confirm that the notebook run
succeeded and was evaluated.

This means the report records the public score, but does not invent a strict audit result
for c8-r2.

#### Public result and decision

Public score: **2.45**.

**Rejected.** The schedule fix solved the known four-batch feasibility problem, but the
public score still fell below 3.40. c8 should not be promoted based on the earlier invalid
9.05 offline run or on the c8-r2 schedule correction alone.

---

## 5. Engineering work performed after the baseline

The experiments were accompanied by targeted infrastructure work. These changes were
intended to make failures visible and submissions reproducible, not to make the model
larger or train new weights.

### 5.1 Strict Flash artifact auditor

`scripts/audit_flash_output.py` was expanded to validate:

- expected game count and pinned game order;
- expected concurrency and runtime;
- expected analyzer timeout;
- full-run rejection of cancelled games;
- optional single-game targeting for diagnosis;
- post-setup deadline markers;
- terminal grace configuration;
- teardown identity and GPU/process gates;
- preserved score/parquet/metrics/model artifacts and digests;
- watchdog status;
- serving model/tuning/provenance identity;
- optional reasoning-state overlay source/patch digests and import evidence.

The auditor is intentionally read-only. A passing local audit proves the downloaded
artifact satisfies the declared contract; it does not prove a public score improvement.

### 5.2 Teardown hardening

The Flash serving teardown patch captured final metrics once, waited for delayed GPU
release, and retained bounded process identity, port, GPU-query, metric-preservation, and
artifact-preservation gates. This addressed the full-v2 stale-GPU failure and was present
in the clean baseline/candidate bundles.

### 5.3 Runtime and deadline controls

The package builder gained explicit controls for:

- game count and pinned game ID;
- concurrency;
- runtime seconds;
- analyzer timeout;
- terminal grace;
- deadlines measured after model setup for isolated diagnostics;
- batch-aware full-runtime feasibility.

The important correction was to account for serial worker batches, not only the nominal
per-game cap.

### 5.4 Digest-bound state overlay

The state experiment used a source-digest-bound patch rather than an unconstrained edit
of the vendored agent. If the pinned tool-agent source changed, the patch refused to
apply. The output recorded both source and patched hashes. This made the state-overlay
experiment auditable, even though its public score was poor.

### 5.5 Local test/check history

The relevant implementation work was checked incrementally:

- the baseline audit/package work reached a passing focused/full test state;
- the c12/targeted-runtime work reached **55 passed, 2 subtests passed** in the recorded
  handoff;
- c8-r2 schedule work reached **63 passed, 1 skipped, 15 subtests passed**;
- `git diff --check` was used to catch formatting issues before package generation.

These are software checks. They did not consume GPU gameplay and do not establish a
public score.

---

## 6. What was learned

### 6.1 Offline score is a weak promotion signal

The major examples are:

| Variant | Offline mean | Public score | Interpretation |
|---|---:|---:|---|
| Baseline c28 | 5.502654474314745 | **3.40** | incumbent |
| State-overlay c12 | 7.0094912357041395 | **1.80** | strong offline, poor public transfer |
| Wuliao/Tufa reproduction | 6.26606582050872 | **2.54** | strong offline, still worse public |
| Original c8 full | 9.05124079307816 | — | invalid because one game cancelled |
| Full v2 rerun | 5.6051924776763435 | — | invalid because teardown failed |

The offline score is useful for finding regressions and resource effects, but it is not a
reliable ordering of public candidates in this history.

### 6.2 Runtime validity is part of model quality

The original c8 run looked like the best offline result until the cancelled `tr87` game
was noticed. The full-v2 run looked slightly better than baseline until the stale GPU
process invalidated teardown. These were not cosmetic problems: they changed whether the
result was a reproducible, complete experiment.

### 6.3 Lower concurrency is not automatically better

The rationale for c8/c12 was credible: the vLLM profile retained eight sequences and the
baseline logs showed severe KV-cache pressure, analyzer timeouts, queued requests, and
preemption. However:

- c8 introduced four-batch scheduling pressure;
- the first c8 full run cancelled the final game;
- c8-r2 fixed schedule feasibility but scored 2.45 publicly;
- state-overlay c12 scored 1.80 publicly;
- no-overlay c12 scored 2.81 publicly.

Concurrency-only changes did not produce a public improvement.

### 6.4 A technically valid experiment can still be a bad submission

The state-overlay c12 full run passed the strict offline audit. That was necessary but not
sufficient. The public score of 1.80 shows why a clean artifact gate and a public score
gate must both exist.

### 6.5 The task was inference/runtime work, not training

The repository packages a pinned external Qwen3.8 Flash-Next NVFP4 model and the Duck
inference harness. The post-3.40 work changed serving configuration, agent patching,
concurrency, deadlines, teardown, and packaging. No successful weight-training experiment
was performed, and no Slurm training job was required for these hypotheses. Using the
university server would not have addressed the observed public-score gap unless a new
trainable model or a separately justified offline-training experiment were introduced.

---

## 7. Current decision and submission posture

### Keep

- submission `56170646` as the protected incumbent;
- the baseline notebook/runtime profile as the default submission;
- the strict audit and batch-aware schedule checks;
- the teardown hardening and reproducibility records.

### Do not promote

- full-v2: failed teardown;
- original c8 full: cancelled final game;
- state-overlay c12: public 1.80;
- Wuliao/Tufa reproduction: public 2.54;
- no-overlay c12: public 2.81 and no independently available strict artifact audit;
- c8-r2: public 2.45 and inaccessible exact strict-audit artifact.

### Recommended stopping rule

Pause concurrency-only variants. A new submission should require a concrete mechanism that
could plausibly change public behavior, not just a different c8/c12 setting or a higher
offline mean. The next candidate should:

1. state one falsifiable mechanism-level hypothesis;
2. run a clean preflight with the exact intended configuration;
3. complete and pass a fresh 25-game full audit;
4. submit only the completed audited notebook version;
5. replace 3.40 only if the public score is strictly higher.

---

## 8. Evidence index

### Repository artifacts

- Baseline source: `arc-agi-flash-next-mtp.ipynb`
- Builder: `scripts/build_flash_next_package.py`
- Auditor: `scripts/audit_flash_output.py`
- State patch: `scripts/flash_agent_state_patch.py`
- No-overlay c12 handoff: `kaggle/flash-next-nostate-c12/README.md`
- c8-r2 handoff: `kaggle/flash-next-c8-r2/README.md`
- Saved outputs: `build/kaggle-output/`

### Saved run directories

```text
build/kaggle-output/flash-full-v1/
build/kaggle-output/flash-full-v2/
build/kaggle-output/flash-preflight-c8-v1/
build/kaggle-output/flash-c8-full-v1-20260914/
build/kaggle-output/flash-c8-tr87-preflight-v1-20260914/
build/kaggle-output/flash-c8-tr87-preflight-v2-20260915/
build/kaggle-output/flash-state-c12-preflight-v2/
build/kaggle-output/flash-state-c12-full-v1/
build/kaggle-output/flash-preflight-c28-control1200-v1/
build/kaggle-output/flash-preflight-c12-candidate1200-v1/
build/kaggle-output/flash-c28-short-control25-v1/
build/kaggle-output/flash-c12-short-deadline25-v1/
build/kaggle-output/wuliao-anim-repro-v1-20260915-2059/
```

The c8-r2 **package** is saved at:

```text
build/flash-c8-full-r2-20260918/
```

The exact remote c8-r2 output is not currently available through the owner credentials,
so the package directory must not be confused with a downloaded strict-audit artifact.

### Relevant implementation commits

| Commit | Purpose |
|---|---|
| `08b9e05` | Flash-Next package, audit, and teardown infrastructure |
| `0389e13` | Targeted game selection, runtime/concurrency controls, stricter audit/deadline checks |
| `c8deb63` | Experimental no-overlay c12 handoff and digest-bound state-patch support |
| `f1b15a9` | Quota-safe c8-r2 package and full-runtime batch feasibility guard |

### Public submissions

```text
56170646  3.40  Flash-Next MTP; audited public25; GPU-release teardown fix
56249553  1.80  Flash-Next MTP state-c12; audited public25
56265695  2.54  Wuliao/Tufa exact public-source reproduction; audited 25-game run
56310095  2.81  direct no-overlay c12 submission
56338677  2.45  c8-r2 submission
```

---

## Final conclusion

The post-3.40 search tested reliability reruns, lower concurrency, targeted deadline
behavior, persistent reasoning state, a public-source reproduction, batch-aware scheduling,
and a no-overlay control. The work improved the packaging and audit discipline, but none
of the tested mechanisms improved the public score.

**The correct submission decision remains: keep the 3.40 Flash-Next/MTP baseline.**


## 9. Continuation: ACTION7 passthrough c28 (2026-09-19)

**Status at launch: locally verified; later completed and rejected after public scoring.**
The 3.40 incumbent is unchanged. This entry is appended without rewriting the
historical experiments above.

- Mechanism: the pinned Flash adapter advertises `ACTION7` but its reverse
  mapping returns `None`. Add only `"ACTION7": "ACTION7"` to that mapping.
- Evidence: `scripts/diagnose_flash_action7.py build/kaggle-output/flash-full-v1`
  reports 33 tool snippets mentioning ACTION7 across six games, zero engine
  executions. These are snippet mentions, not exact attempted-action counts.
- Unlike the older non-Flash checkpoint-8/UNDO/reset-sync bundle, this retains
  the neutral action name and changes neither reset handling nor memory.
- Retain c28, 7,920 seconds/game, analyzer timeout 900 seconds, serving profile,
  prompts, sampling, and scorer. No concurrency-only retry or state overlay.
- Local checks: pinned source before/after mapping regression, actual pinned
  solver dispatch with a stub engine (including unavailable-action rejection),
  generated overlay/import tests, and artifact-audit rejection tests.
- Re-audited the existing baseline with real Parquet validation: passed, 25 games,
  offline mean 5.502654474314745, 4,683 actions, zero restarts, clean teardown.
- Remote preflight `izukia/arc-agi3-flash-next-mtp-action7-preflight25/1` and
  full kernel `izukia/arc-agi3-flash-next-mtp-action7-full/1` later completed.
  The strict preflight audit passed with offline mean **2.0970268186983443**;
  the strict full audit passed with offline mean **5.975441389862917** and
  3,791 actions. The public submission `56369560` scored **2.32**, below 3.40.
- New packages: `build/flash-action7-preflight25-20260919/` and
  `build/flash-action7-full-20260919/`. These are packages, not completed runs.
- Reproduction and strict gates: `kaggle/flash-next-action7/README.md`.

**No score improvement is established.** The completed audits establish runtime
and artifact evidence only. Enabling a previously broken action can still hurt
action efficiency; the public result confirms that this candidate should not be
promoted.


## 10. Inspection-action follow-up v3 preflight (2026-09-22)

**Status: rejected; no public submission.** This entry records the completed v3
artifact separately from the v4 full candidate now waiting in Kaggle's queue.

- Mechanism: after an inspection-only Python call, force one bounded follow-up
  asking the model to emit an environment action. The v3 implementation forced
  this after ordinary inspection calls as well as timeout-stalled calls.
- Remote kernel: `izukia/arc-agi3-flash-next-mtp-inspect-preflight/1`.
- Artifact: `build/kaggle-output/flash-inspection-action-preflight25-20260922-v1/`
  (the same downloaded output was also preserved under the `-v2` download folder).
- Scope: 25 public games, c28, 1,800-second preflight budget, analyzer timeout
  900 seconds, baseline serving/model/scorer otherwise unchanged.
- The completed artifact passed the then-current structural/provenance audit:
  25/25 games, clean teardown, zero watchdog restarts, and 720 actions.
- Offline mean: **1.3427190620328542**. The run forced 115 inspection follow-ups
  and recorded three exhausted follow-up turns. This is below the clean c28
  25-game 1,800-second control (**1.4458934228164997**) and far below the
  full-runtime baseline offline mean (**5.502654474314745**; not an apples-to-apples
  runtime comparison).

**Decision: reject.** The policy changed ordinary solver behavior too broadly and
  did not improve the protected baseline. Keep the artifact for failed-experiment
  provenance; do not submit it.


## 11. Inspection-action follow-up v4 direct full run (2026-09-22)

**Status: queued; no result or public submission.**

- v4 narrows the hypothesis: force one follow-up only when an inspection-only
  Python call reaches the `turn_time_budget`; ordinary inspections retain baseline
  behavior.
- Patch source digest:
  `535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d`.
- Patched tool-agent digest:
  `c8e3132374da9d333656114557d9bf7d44dc24e088522ec2668c0fddd5396563`.
- Full kernel: `izukia/arc-agi3-flash-next-mtp-inspect-full/1`.
- The remote notebook's 18 cell sources match the local v4 package exactly.
- The full candidate retains c28, 7,920 seconds/game, analyzer timeout 900
  seconds, the baseline serving/model/scorer, and no ACTION7, direct-retry, or
  state-overlay patch.
- As of the latest continuation check it remains `QUEUED`; GPU quota is not the
  limiting factor. Download to a fresh output directory and require a strict
  25-game full-offline audit before considering any submission.

The protected incumbent remains submission `56170646` at **3.40**.


## 12. Score-sorted public-code audit and current Kaggle reconciliation (2026-09-24)

This continuation reconciles the ledger against the live Kaggle code list, current
submission list, and the previous-session public-code handoff. It does not replace the
historical 2026-09-19 scoreboard above.

### Current live submissions

The live submission listing currently shows:

```text
56514817  PENDING   Flash-Next MTP exact 3.40 baseline replay; strict audited public25; c28
56488018  COMPLETE  2.80  Flash-Next MTP inspection-action follow-up v4
56369560  COMPLETE  2.32  Flash-Next MTP ACTION7 passthrough
56170646  COMPLETE  3.40  Flash-Next MTP protected incumbent
```

Submission `56514817` was accepted despite the CLI's `0 submissions remaining today`
message, but it has not received a public score in the current check. It therefore does
not change the incumbent or establish a replay result. Submission `56488018` is no
longer pending; its public score is **2.80**. ACTION7 is no longer pending either; its
public score is **2.32**. Promotion still requires a completed public score strictly
above **3.40**.

The completed v4 artifact remains useful offline evidence: 25/25 games, c28, 7,920
seconds/game, analyzer timeout 900 seconds, exact v4 patch identity, zero watchdog
restarts, `shutdown_ok=true`, offline mean **6.22080236423742**, and 4,937 actions.
Those facts do not override the public score of 2.80.

Two additional kernels completed and passed fresh strict audits in this continuation:

| Kernel/artifact | Strict audit result | Offline mean | Submission decision |
|---|---|---:|---|
| `izukia/arc-agi3-flash-next-mtp-c28-replay/1` / `build/kaggle-output/flash-c28-replay-20260924-v1/` | 25 games, 4,200 actions, clean teardown | 6.638753065395086 | Submitted as pending probe `56514817` |
| `izukia/arc-agi3-flash-next-mtp-zc-guard/1` / `build/kaggle-output/flash-zc-guard-20260924-v1/` | 25 games, clean teardown | 4.55674837292894 | Not submitted; worse than paired replay and no public evidence |

Both audits used the exact downloaded kernel-log name. The replay's offline mean is not
a public score and does not establish an improvement while `56514817` remains pending.

### Public code-list filter and source inspection

The live score-sorted code list was inspected with authenticated Kaggle metadata and
source pulls. The following high-score entries were independently compiled or source-
compared locally; no competition submissions were made from them:

| Notebook | Displayed best public score | Local finding | Decision |
|---|---:|---|---|
| `chiakazirim/duck-qwen3-8-tuned` | 4.50 | Public fork; only three source-line changes from the Flash family | Control/lead, not yet run |
| `wuliao0/duck-qwen3-8-anim-base` | 4.33 | Source-identical Flash-Next/MTP family | Reproduction control |
| `amanatar/arc-agi-3-hybrid-repl-agent` | 4.08 | Genuine serving-profile variant; scored version 15 differs from current version 38 | Highest-risk distinct lead |
| `juliancamilovilla/arc-agi3-nvfp4` | 3.95 | Flash-family source with a bounded offline-cut change | Not run |
| `tantan0327/arc3-flashnext-asis` | 3.92 | Source-identical Flash-family notebook | Control |
| `yanggod/arc-agi-3-duck-flash-next-mtp` | 3.71 | Explicitly an unchanged fork of Keith's Flash notebook | Control |
| `rakbidb/sukirman-verified-exact-profile-flash-next-mtp3` | 3.48 | Gameplay and serving unchanged; adds exact-profile checks/metrics | Reliability control |
| `fanglimo/arc3-flash-next-full` | 3.44 | Adds pinned-input checks and teardown hardening; no gameplay change | Reliability control |

The code page explicitly labels `ghazarosghazaros/duck-qwen3-8-anim-base` and
`muhibullahansir/duck-qwen3-8-anim-base` as **“Notebook copied with edits from a
private notebook.”** Both were excluded despite displayed scores of 4.09 and 4.16.
The previous-session handoff also recorded that the high-scoring Anim entries largely
share the same Flash-Next source, so their displayed scores are not evidence of a new
solver mechanism.

All pulled candidate scripts/notebooks in this audit passed local Python or notebook
cell compilation. This is source validation only: no candidate above received a Kaggle
Save & Run result in this continuation.

### Amanatar v15 preflight attempt

The exact public score of **4.08** belongs to Amanatar `scriptVersionId=347399496`,
kernel version 15. The current public notebook version 38 is a different version and
has an error warning, so version 15 was recovered directly before preparing the test.

- Candidate package: temporary `/tmp/arg-agi3-amanatar-v15-preflight/`.
- Inputs: the public Flash-Next model/runtime/source datasets, GPU enabled, internet off.
- Test-only changes: one offline public game, one pass, a 1,500-second per-game cap,
  and a private candidate slug. The protected baseline was not modified.
- Local result: source compilation passed.
- Upload result: repeated `kaggle kernels push` attempts were rejected by Kaggle's
  `SaveKernel` endpoint with HTTP **403 Forbidden**, including after OAuth reauth. A
  direct web-service route returned **404** and did not create a kernel.
- Ownership check: `izukia/arc-agi3-amanatar-hybrid-v15-preflight` was not found after
  the attempts. No remote execution, GPU quota consumption, or competition submission
  occurred.

**Decision:** retain Amanatar v15 as an untested public lead. Do not treat its displayed
4.08 as reproduced, and do not promote it over submission `56170646`.

### Updated conclusion

The current evidence still supports the same operational decision: keep the protected
Flash-Next/MTP **3.40** incumbent. The public-code audit found useful controls and one
distinct Amanatar serving lead, but no new candidate has passed the complete
build/preflight/full-audit/public-score chain.
