# Dyna-WM baseline — ARC-AGI-3

A legitimate, observation-only model-based RL agent. It learns each game from
real interaction during play. It does **not** read the hidden game source.

## Why not the v31 "0.36" approach

The v31 reference scores by reading the hidden game's `.py` file off the Kaggle
filesystem, regex-extracting the exact win condition, and importing + deep-copying
the real game object to simulate it perfectly. That is reading the answer key, not
generalizing. ARC-AGI-3 exists to measure generalization to novel unseen games, so
that route circumvents the task, risks disqualification, and breaks the moment the
organizers sandbox the game source. We reuse v31 only for the toolkit interface
(`Agent` subclass, `arcengine` action calls, Kaggle scaffolding).

## The idea, in one line

Two cooperating models: a **world model** that learns to *see* the game (predict
what happens next), and a **DQN** that learns to *win* (maximize reward), with the
world model's prediction error doubling as a curiosity signal that drives
exploration.

## Components

| Piece        | Input → Output                                  | Role |
|--------------|-------------------------------------------------|------|
| `Encoder`    | grid (18×64×64) → latent `z` (128) + spatial feat | shared perception |
| `WorldModel` | (`z`, action) → next `z`, predicted reward      | model 1 — "sees the game" |
| `q_discrete` | `z` → Q-values for ACTION1..5                   | model 2 — "wants to win" |
| `ClickHead`  | spatial feat → 64×64 Q-map for ACTION6          | model 2 — spatial wins |
| Curiosity    | ‖predicted `z` − actual `z`‖² → intrinsic reward | exploration drive |

Observation = 16 colour one-hot planes + 2 coordinate planes. Actions follow the
ARC-AGI-3 space: ACTION1–4 (directions), ACTION5 (interact), ACTION6 (x,y click),
ACTION7 (undo), RESET.

## Training — Dyna style

Every few env steps we sample a batch from replay and compute three losses:

1. **Real TD loss.** Standard DQN target on actually-observed transitions, over the
   *available* action set (discrete and click masked to what the frame allows). A
   soft-updated target network stabilizes the bootstrap.
2. **World-model loss.** Supervised next-latent + reward prediction on real steps.
   This is what makes model 1 accurate.
3. **Imagined Dyna loss.** Short `K`-step latent rollouts (`K=3`) branched from real
   encoded states, training the DQN on transitions the world model *imagines*. This
   is the sample-efficiency multiplier — many cheap gradient updates per scarce real
   step.

Curiosity: the world model's latent prediction error is added to the reward as an
intrinsic bonus (`BETA_INT`), pushing the agent toward states the model can't yet
predict — i.e. where there's something new to learn.

## Current LS20 planner baseline

For the local `ls20` experiment we currently use a pretrained observation-only
world model in `ls20_model.pt` plus two lightweight planners:

- `bfs`: beam search over imagined next states from the world model.
- `mcts`: UCB-style Monte Carlo tree search over the same imagined transitions.
- `both`: runs both planners, prints the comparison, and chooses an imagined win
  first; otherwise it chooses the higher-scoring plan.

By default `ls20_agent.py` runs the world model frozen. That keeps local tests
cheap and avoids spending time fine-tuning on every probe. Add `--online-train`
only when you intentionally want to keep updating the world model while playing.

Recommended cheap local run:

```bash
/home/izu/Projects/.venv/.venv-310/bin/python ls20_agent.py \
  --model ls20_model.pt \
  --planner both \
  --games 1 \
  --max-steps 100 \
  --depth 5 \
  --beam 64 \
  --mcts-sims 64 \
  --commit-steps 2 \
  --out recordings/agent_replay.npz
```

Replay the saved run:

```bash
/home/izu/Projects/.venv/.venv-310/bin/python replay_ls20.py \
  --model ls20_model.pt \
  --npz recordings/agent_replay.npz \
  --out replays/ls20_comparison.gif
```

## Kaggle submission path

The Kaggle notebook is generated from `_build_nb.py`. It inlines `cwm.py` into
`my_agent.py`, writes `/kaggle/working/my_agent.py`, copies it into the official
`ARC-AGI-3-Agents` harness, and runs `main.py --agent myagent` during competition
reruns.

Current notebook defaults:

```bash
ARC_PLANNER=bfs
ARC_BFS_DEPTH=3
ARC_BFS_BEAM=8
ARC_MCTS_DEPTH=3
ARC_MCTS_SIMS=32
ARC_MCTS_CPUCT=1.4
ARC_CWM=/kaggle/input/forge-pretrained-weights/cwm.pt
```

Planner options in `my_agent.py`:

- `value`: cheapest one-step CWM lookahead.
- `bfs`: default submission planner, shallow beam search over imagined states.
- `mcts`: UCB search over imagined states; slower but now runnable locally and in
  the generated notebook.

Local planner comparison:

```bash
/home/izu/Projects/.venv/.venv-310/bin/python compare_planners.py \
  --game ls20 \
  --planners value bfs mcts \
  --max-actions 50 \
  --bfs-depth 3 \
  --bfs-beam 8 \
  --mcts-sims 32
```

### The deliberate safety choices

- **Short imagined rollouts (K=3), not long ones.** A DQN's `max` exploits model
  error; long rollouts from a still-weak model train toward hallucinations. Short
  rollouts cap how far error compounds.
- **Imagined loss down-weighted** (`IMAGINE_W=0.5`) relative to real TD loss, so
  real data dominates while the model is unreliable.
- **Imagined rollouts use discrete actions only.** The 4096-way click space is too
  large to imagine reliably; clicks are learned from real transitions only.

## Key constants (top of `my_agent.py`)

```
D=128  GAMMA=0.99  TAU=0.01  BETA_INT=0.5  LR=3e-4
BATCH=64  TRAIN_EVERY=4  IMAGINE_K=3  IMAGINE_W=0.5
```

## Honest limitations

- **From-scratch online learning is hard here.** ARC-AGI-3 gives a limited action
  budget per novel game; a DQN learning from zero may not converge before the budget
  runs out. The world model + imagined Dyna improves sample efficiency but does not
  eliminate this risk. Frontier systems currently score near zero on these games.
- **Next step if scores are low:** pre-train the encoder + world model offline on the
  public games to learn general dynamics priors, then adapt online per hidden game.
  The architecture supports this — only the weight-loading path needs adding.
- **If the separate DQN and world model fight**, the proven consolidation is
  DreamerV3 / EfficientZero, which fold value learning *into* the world model rather
  than keeping a standalone DQN.

## Files

- `my_agent.py` — the agent (verified: compiles, forward/backward pass, all action
  branches exercised by `_smoke.py`).
- `ls20_agent.py` — local LS20 world-model planner with `bfs`, `mcts`, and `both`
  modes.
- `compare_ls20_wm.py` — compares the LS20 world model's predicted next state
  against real engine observations.
- `arc-agi.ipynb` — submission notebook: install wheels → write agent → wire into the
  competition harness on rerun → local submission stub.
- `_smoke.py` — offline test harness (stubs `arcengine`/`agents.agent`).
