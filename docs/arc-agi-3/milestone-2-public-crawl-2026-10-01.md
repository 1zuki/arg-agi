# ARC-AGI-3 Milestone 2 Public Crawl

**Crawl date:** 2026-10-01
**Competition:** [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
**Scope:** public code leaderboard, Milestone 2 notebooks, public write-ups, and the relevant Kaggle discussions.
**Local reference:** [`FLASH_EXPERIMENTS_SINCE_3_40.md`](../../FLASH_EXPERIMENTS_SINCE_3_40.md)

## Executive summary

Milestone 2's public leaders are not using a fundamentally different solver family. The top three all use the **Tufa Labs Duck/TAAF harness**, **Qwen3.8-Flash-Next**, an RTX PRO 6000, and speculative decoding.

The large public gap is mainly inference engineering:

- FP8 KV cache instead of a smaller or more restrictive cache profile;
- roughly one million or more retained KV tokens;
- much longer rolling histories;
- prefix-cache and Mamba-state retention;
- cache-aware admission and blockwise context trimming;
- serving and scheduling changes that keep useful game state resident;
- harness corrections for stale states, animations, terminal frames, tools, and level handover.

The clearest reported ablation is from rellik13: KV occupancy had reached 91–97%, requests were queued waiting for cache space, and moving to FP8 KV plus a larger history window reportedly moved the score from about **14.5 to 22.5** without a prompt change.

The practical conclusion for this project is to investigate **memory capacity and state retention before prompt engineering**. Our protected 3.40 submission is still the promotion gate; no public result from this crawl changes that incumbent decision automatically.

## Public score comparison

| Entry | Public score | Delta vs our 3.40 | Delta vs TAAF FlashNext 6.09 | Evidence |
|---|---:|---:|---:|---|
| dfranzen | **27.89** | **+24.49** | **+21.80** | Public leaderboard and winner comparison |
| lordhansolo | **23.84** | **+20.44** | **+17.75** | Public leaderboard and winner comparison |
| sirikilohit / rellik13 | **22.53** | **+19.13** | **+16.44** | Public leaderboard and winner comparison |
| richardcsaky | **20.00** | **+16.60** | **+13.91** | Public notebook page |
| scottlegrand TAAF FlashNext | **6.09** | **+2.69** | — | Public notebook page |
| Our protected Flash-Next/MTP `56170646` | **3.40** | — | **-2.69** | Local ledger plus Kaggle result |

These are leaderboard values from public competition evaluations. Scores from different submission windows are not controlled experiments, and the competition has substantial single-play variance.

## Winner configuration comparison

The following values come primarily from the public comparison post `744792` and the linked write-ups.

| Property | dfranzen | lordhansolo | rellik13 / sirikilohit | Our protected 3.40 |
|---|---|---|---|---|
| Target model | Intel W4A16 AutoRound | Mixed NVFP4/FP8 with BF16 PLE | Intel W4A16 AutoRound | Pinned NVFP4 Flash-Next |
| Draft / MTP | Albucino INT4 MTP, 3 speculative tokens | Built-in NVFP4 MTP | RadixArk NVFP4 MTP, 3 speculative tokens | 3 MTP tokens |
| Server | SGLang Pennyroyal 2.5.3 | vLLM nightly 0.29.1rc1, commit `e975732` | SGLang Pennyroyal 2.5.0 | Pinned vLLM runtime |
| KV dtype | FP8 E4M3 | FP8 E4M3 | FP8 E4M3 | Prefix caching disabled; smaller pinned profile |
| KV pool | 1,011,264 tokens | 1,417,100 tokens | 1,004,288 tokens | Effective vLLM limit of 8 sequences; no million-token pool |
| Context | 139,264 server / 131,072 harness | 147,072 server / 127,488 harness | 69,632 | 65,536 server / 32,768 analyzer |
| Live streams | 10 | 14 | 16 | Duck concurrency 28, but vLLM effectively serves 8 |
| State retention | Prefix cache, sparse Mamba checkpoints, large hysteresis trim | Prefix cache, FP8 indexer KV, BF16 Mamba cache | Longer history, hierarchical KV/Mamba host tier | Conservative pinned serving profile |
| Scheduling | Priority admission, tail fade, 59k-token drain | 3,918-second game cap and 8-wave admission | 2,400-second slices plus UCB allocation | Existing Duck scheduling and fixed profile |

The leaders trade nominal concurrency for **cache-resident useful state**. Our concurrency number is therefore not directly comparable to their live-stream count: the serving limit and cache pressure determine how much work can proceed without eviction or queuing.

## dfranzen: highest public result

Sources: [repository](https://github.com/da-fr/arc-agi-3-solution), [write-up](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md), [Kaggle notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution).

### Serving profile

- Intel W4A16 AutoRound target model;
- Albucino INT4 MTP draft;
- SGLang Pennyroyal v2.5.3;
- FP8 E4M3 KV cache;
- about 1.01M KV tokens;
- 139k server context and 131k harness context;
- 10 active streams and 60 Mamba cache slots;
- about 1.01M total KV tokens, with roughly 1.14GB of CUDA graph memory and 4.25GB free GPU memory in the published comparison.

### Harness and scheduling changes

- animation frames and timelines;
- 10x image scaling and difference images;
- fatal/game-over frames;
- `UNDO` support;
- persistent model-defined Python functions;
- image-aware token counting;
- stale-state, no-op, and terminal guards;
- improved level-transfer and new-level prompts;
- priority scheduling with a cache-aware admission gate;
- large blockwise trimming, approximately 118k down to 59k;
- Mamba checkpoint retention and prefix-cache reuse;
- prefetch and kernel/runtime fixes around small-batch serving.

The write-up reports that structured world-model summaries and periodic summarization did not help. A larger rolling context did.

## lordhansolo: vLLM variant

The public comparison gives lordhansolo a score of **23.84**. The profile is notable because it reaches the top group with vLLM rather than SGLang:

- mixed NVFP4/FP8 target with BF16 PLE;
- built-in NVFP4 MTP;
- vLLM nightly `0.29.1rc1`, commit `e975732`;
- FP8 KV cache and approximately 1.417M KV tokens;
- 147k server context and 127k harness context;
- 14 live streams;
- prefix caching, FP8 indexer KV, and BF16 Mamba cache;
- measured profile with approximately 19.69GB of KV plus Mamba cache on GPU.

The notebook's current offline diagnostics should not be substituted for its public 23.84 result.

## rellik13 / sirikilohit: memory ablation evidence

Sources: [repository](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution), [write-up](https://raw.githubusercontent.com/LohitSiriki/arc-agi-3-milestone2-solution/main/WRITEUP.md), [Kaggle notebook](https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit).

### Published configuration

- Intel W4A16 target with RadixArk FP8 PLE;
- RadixArk NVFP4 MTP draft;
- SGLang Pennyroyal v2.5.0;
- FP8 KV cache;
- approximately 1.004M KV tokens;
- 69,632-token context;
- 16 concurrent requests;
- history trim changed from approximately `37k -> 27k` to `57k -> 45k`;
- 2,400-second UCB scheduler slices;
- 48GB host tier for hierarchical KV and Mamba state.

### Direct discussion evidence

Rellik13 reported that the KV pool sat at 91–97% full and requests queued for cache space. The memory limit, rather than decode throughput, was the bottleneck. FP8 KV freed memory for history, and the larger history window was credited with the jump from roughly 14.5 to 22.5.

This is the strongest public causal clue in the crawl, although it is still a self-reported ablation from a stochastic competition run rather than a replicated controlled experiment.

## richardcsaky: a distinct runtime package

The public notebook reports **20.00**. Its package is materially different from the simple Flash profile:

- expert overlay checkpoint;
- vLLM serving with prefix caching;
- `--max-model-len 262144`;
- `--max-num-seqs 35`;
- int4 per-token-head KV settings;
- Marlin backend and asynchronous scheduling;
- additional expert-pruning and serving patches.

The notebook also includes a separate public-validation artifact and should be treated as a lower-confidence comparison than the three-way winner table. Its public score is evidence; its local preview metrics are not a replacement for that score.

## What the controlled discussions say

### Harness or model? (`743723`)

This study held the model, serving stack, and time budget fixed and added harness changes one at a time. Its main findings were:

- identical builds varied substantially between runs;
- two immediate identical runs scored 2.60 and 1.45;
- full runs of one build ranged roughly 4.5–9.4;
- consensus, reasoning-discipline blocks, higher-resolution images, and executable world-model scaffolds were neutral or negative in that setup;
- many stuck runs had already tried the decisive action type but committed to the wrong goal hypothesis early;
- a single run cannot reliably resolve small effects.

This argues against attributing the Milestone 2 gap to a simple prompt addendum. The leaders' serving changes may be what lets a good hypothesis survive long enough to pay off, while the underlying model still determines whether the initial goal judgement is correct.

### Perception-augmented Duck (`743060`)

This independent write-up used Qwen3.8-27B FP8 on the Duck harness and reported controlled public/hidden results:

- deterministic connected-component click hints were kept;
- two scoring and budget prompt lines were kept;
- 32k to 64k context was rejected after throughput collapsed;
- temperature 0.6 to 0.3 was rejected because exploration collapsed;
- an archetype playbook was rejected because it caused premature classification.

The absolute scores in this study are not comparable to the Milestone 2 winner scores, but the failure mechanisms are useful: more context only helps when the serving profile can sustain it, and more explicit strategy can lock the model into a wrong hypothesis.

## Our local evidence

The local ledger records the protected incumbent and several failed transfer attempts:

| Local variant | Offline mean | Public score | Interpretation |
|---|---:|---:|---|
| Protected Flash-Next/MTP c28 | 5.5027 | **3.40** | Incumbent |
| State-overlay c12 | 7.0095 | **1.80** | Strong offline result, poor public transfer |
| Wuliao/Tufa reproduction | 6.2661 | **2.54** | Strong offline result, still below incumbent |
| No-overlay c12 | — | **2.81** | Rejected |
| c8-r2 | — | **2.45** | Rejected |

The local record therefore supports three rules:

1. Offline means are useful for diagnosing regressions and resource effects, not for promotion.
2. Lowering concurrency alone is not a demonstrated public improvement.
3. Promotion requires a complete audit and a public score strictly above **3.40**.

## TL;DR: them vs ours vs TAAF FlashNext

### The leaders versus ours

- Same broad solver lineage, radically different serving envelope.
- They retain roughly **1.0–1.4M KV tokens**; ours has an effective eight-sequence vLLM limit and prefix caching disabled.
- They use FP8 KV, long histories, Mamba/prefix state retention, cache-aware admission, and aggressive trims.
- Our previous experiments changed concurrency and harness behavior without reproducing their memory profile, so they do not test the same hypothesis.
- The first high-value replication target is **FP8 KV plus larger retained history**, not a new prompt.

### The leaders versus TAAF FlashNext

- TAAF FlashNext is already closer in lineage to the winners than a generic baseline, but its public 6.09 is still far below 20–28.
- The winners appear to be TAAF/Duck plus a much more capable memory and serving configuration.
- The public evidence points to inference engineering as the dominant Milestone 2 differentiator, with model judgement and run variance still limiting the ceiling.

### Our immediate experiment order

1. Add an FP8-KV serving branch with explicit cache-pool occupancy telemetry.
2. Sweep history retention and trim hysteresis, including the reported `57k -> 45k` range.
3. Preserve prefix and Mamba state across turns where the runtime supports it.
4. Add cache-aware admission and context-size guards before increasing concurrency.
5. Compare SGLang and vLLM only after matching the memory budget and retained history.
6. Revisit prompt/perception changes after the serving profile is no longer memory-starved.

No new Kaggle run is implied by this report. Any candidate must pass the existing package, full offline audit, teardown, and public-score gates before it can replace submission `56170646`.

## Sources

- [User-supplied Milestone 2 code leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code?competitionId=133468&sortBy=scoreDescending&excludeNonAccessedDatasources=true)
- [User-supplied discussion index](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion?sort=published)
- [Comparing the Milestone 2 winning solutions (`744792`)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)
- [Harness or model? (`743723`)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723)
- [Perception-Augmented Duck (`743060`)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060)
- [dfranzen repository](https://github.com/da-fr/arc-agi-3-solution)
- [dfranzen write-up](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md)
- [rellik13 repository](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution)
- [rellik13 write-up](https://raw.githubusercontent.com/LohitSiriki/arc-agi-3-milestone2-solution/main/WRITEUP.md)
- [richardcsaky notebook](https://www.kaggle.com/code/richardcsaky/arc-agi-3-milestone-2-submission)
- [scottlegrand TAAF FlashNext notebook](https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922)
- [sirikilohit notebook](https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit)
- [lordhansolo notebook](https://www.kaggle.com/code/lordhansolo/arc-agi-3-milestone-2)
- [dfranzen notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
