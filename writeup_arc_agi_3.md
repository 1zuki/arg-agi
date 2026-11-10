# How We Went from a World Model to a Duck

## A second-year team’s ARC-AGI-3 journey

- **Authors:** A two-person team of second-year students
- **Track:** ARC-AGI-3
- **Code submission:** [combined Milestone 2 profile](https://www.kaggle.com/code/izukia/arc-agi3-m2-combined-profile-v1), submission `57385516`

### TL;DR

- We started by trying to learn a world model, then search through imagined futures.
- That approach taught us how to think about state and exploration, but it was too hard to make reliable with our compute budget.
- Milestone 1 pointed us to Tufa Labs’ Duck/TAAF harness, which became the foundation of our later work.
- A Flash-Next run reached **5.96**, around 35th place in our snapshot at the time.
- Our best verified combined-profile submission reached **29.69**. A later full rerun scored **6.92**, so we treat the higher score as a strong result from one run, not a clean estimate of a single improvement.

### We joined because the agent had to learn while playing

ARC-AGI-3 started on March 25, 2026. Instead of giving an agent fixed input-output examples, it puts the agent in an unfamiliar interactive environment. It has to work out what matters, remember what happened, and spend actions carefully. The [official competition description](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) made that sound like a problem we wanted to learn from.

We were a team of two second-year students without a polished research plan. We were excited to try something that felt closer to learning and reasoning than ordinary supervised prediction. Our first idea was ambitious: learn enough of each game to plan inside a learned model.

### First, we tried to learn a world

Our first serious direction was an observation-only Dyna-style agent. We encoded each grid observation into a latent representation, trained a world model to predict the next state, reward, and game status, and used a value or DQN head to choose actions. Prediction error supplied a curiosity signal, so the agent would visit states it did not understand yet.

The hope was to spend real actions collecting transitions, then spend cheap model steps imagining short futures. We tried random and exploratory data collection, one-step value selection, shallow BFS-style beam search, and MCTS over imagined states. Our longer-term idea was to train an RL policy to decide which branch of the search tree deserved more computation.

The idea sounded right on paper. In practice, every part had to learn quickly inside a small per-game action budget. The model was often too uncertain for deeper search, and our hardware made large experiments difficult. We tried a convolutional model, a token-based causal world model, offline pretraining, and imagined rollouts. In a separate experiment, we tried using an LLM to describe the game and guide the lower-level planner. None of these pieces became reliable enough to carry the competition.

We were tired, so we took a break before the first milestone. When we came back, we started looking more closely at the public systems that were already working.

### Milestone 1 gave us a better starting point

The first milestone changed our direction. We read the public solutions and found Tufa Labs’ Duck harness. Their approach was close to the part of our original idea that still felt right: an agent needs a persistent interaction history, useful tools, and a model that can revise its understanding after each action. The difference was that their harness already worked, while we were still trying to make every layer ourselves.

We adopted that direction with explicit credit to [Tufa Labs and Jeroen Cottaar](https://github.com/Tufalabs/duck-harness). This was our first big change in mindset: keep the useful intuition, but build on a system that had survived real interaction with the games.

We tried the Duck approach, made a series of small changes, and kept testing. Our first encouraging result in this phase was the [TAAF Flash-Next/MTP notebook, version 2](https://www.kaggle.com/code/izukia/taaf-flashnext-sheetu12b-offline-test), which scored **5.96** on September 27. In the snapshot we recorded at the time, that was about 35th place. It was the first moment the competition felt tractable. We kept trying to improve it for a while, but eventually got stuck around that level.

### Milestone 2: lock in

Then Milestone 2 arrived. We studied its public solutions, including the [Daniel Franzen Milestone 2 solution and write-up](https://www.kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2), and read the surrounding community discussions. We decided to lock in. The new material gave us a clearer map of where strong systems were getting their advantage. We followed a simple loop: read a public result, identify the mechanism that might explain it, implement the smallest compatible experiment, and submit only when the package and runtime were coherent.

Our combined profile kept the Duck/TAAF interaction pattern and used Qwen3.8-Flash-Next, MTP, Pennyroyal/SGLang, long interaction histories, FP8 KV cache, priority scheduling, and Mamba/prefix-cache support. We also added practical guards around state transitions, resets, output contracts, and package validation. The [public notebook](https://www.kaggle.com/code/izukia/arc-agi3-m2-combined-profile-v1) contains the submitted profile and its attribution.

The score history shows why this competition was difficult to interpret:

| Stage | Submission or evidence | Public score | What it felt like |
| --- | --- | ---: | --- |
| Early local agents | June–July submissions | 0.00–0.88 | The world-model idea was not yet competitive. |
| First promising Flash/MTP line | [TAAF Flash-Next/MTP v2](https://www.kaggle.com/code/izukia/taaf-flashnext-sheetu12b-offline-test) | **5.96** | A stronger model and harness finally moved the score. |
| Combined Milestone 2 profile | `57385516` | **29.69** | Our high point, with the full serving setup in place. |
| Later full rerun | `57910947` | **6.92** | A successful rerun could still regress badly. |

The other combined submissions ranged from 22.62 to 29.69, then later fell to 0.00, 8.98, and 6.92. Participants also discussed sampling variance and public/private discrepancies, including [this variance analysis](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062) and the [public comparison of Milestone 2 solutions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792). That is why we report both the 29.69 result and the later 6.92 rerun. The first is our best verified run; it is not proof that one isolated patch caused the whole jump.

### What stayed with us

The world-model experiments were useful even though they did not become our final solver. They gave us a way to think about hidden state, exploration, imagined futures, and choosing between branches. The Duck/TAAF work gave us a practical way to test those ideas against a real language model and a real serving budget.

The biggest lesson was that an agent’s execution environment matters as much as its model. Context length, cache behavior, scheduling, tool output, resets, and state tracking all decide whether a good idea turns into a useful action. A local score is also only a diagnostic: our offline tests could not predict the competition run, and the 6.92 rerun made that clear.

We learned most from the open-source community. Tufa Labs, Daniel, Kang, and many other participants shared code, measurements, and failed experiments. Their work gave us concrete systems to study when our own plan stalled, and made it possible for a small student team to keep moving.

### Closing

We are grateful that two second-year students had the opportunity to participate in a competition with this level of ambition. We are happy with the progress, even with the regressions and frustrating scoring delays, and happy for everyone who shared work along the way. Our final system came from a long chain of abandoned plans, borrowed ideas, careful attribution, and small pieces of evidence. That chain was the real result for us.

**AI assistance disclosure:** We used Codex/ChatGPT to inspect repository history, organize evidence, and edit this draft. We directed the experiments, reviewed the code and results, and remain responsible for the claims and final submission.
