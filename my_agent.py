# =====================================================================
# CWM-planner agent for ARC-AGI-3
#
# Legitimate, observation-only model-based RL. Does NOT read the hidden
# game source.
#
#   ConvWorldModel  (frozen)  (grid, action, click) -> next grid, reward,
#                             game state            [perception + dynamics]
#   ValueNet        (online)  grid -> V(grid)        ["how close to winning"]
#
# Decision = 1-step CWM lookahead: for every candidate action, imagine the
# outcome and score it
#       score = r_pred + WIN_BONUS * P(win) + GAMMA * V(next_grid)
# then pick the best (epsilon-greedy). The CWM imagination is the
# "wants to win" driver; only the small ValueNet learns online, so it
# adapts within a per-game action budget.
#
# Value training (Dyna): TD on real observed transitions + model-based
# value-iteration backup (max over imagined discrete actions). The frozen
# CWM supplies the dynamics, so V can improve without waiting for real
# reward to propagate.
# =====================================================================
import logging
import os
import random
import time
import traceback
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

from cwm import ConvWorldModel, ValueNet, grids_to_long, G, N_ACTIONS, STATE_NAMES  # NB_STRIP_IMPORT

logger = logging.getLogger(__name__)

KAGGLE_RUNTIME_SECONDS = int(os.getenv("ARC_RUNTIME_SECONDS", str(8 * 3600)))
KAGGLE_SAFETY_SECONDS = int(os.getenv("ARC_SAFETY_SECONDS", "300"))

WIN_IDX = STATE_NAMES.index("WIN")
GAMEOVER_IDX = STATE_NAMES.index("GAME_OVER")

GAMMA = 0.99
WIN_BONUS = 10.0          # value of P(win) in the planning score
LR = 3e-4
TAU = 0.01                # value target soft-update
BUF_CAP = 50000
BATCH = 64
TRAIN_EVERY = 4           # env steps between value-train calls
N_CLICK_CAND = 24         # click positions evaluated per planning step
IMAGINE_W = 0.5           # weight on the model-based value-iteration loss
CURIOSITY = float(os.getenv("ARC_CURIOSITY", "0.5"))  # novelty bonus weight

# planner selection: "value" = 1-step lookahead, "bfs" = beam search
PLANNER = os.getenv("ARC_PLANNER", "value").lower()
# BFS is bounded ON PURPOSE: the CWM is ~96% accurate per step, so error
# compounds geometrically with depth. Depth 3 keeps leaf states ~0.96^3=88%
# trustworthy; the beam prunes the branching so it stays cheap per step.
BFS_DEPTH = int(os.getenv("ARC_BFS_DEPTH", "3"))
BFS_BEAM = int(os.getenv("ARC_BFS_BEAM", "8"))


# ==================== agent ====================
class MyAgent(Agent):
    MAX_ACTIONS = float("inf")
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = (int(time.time() * 1e6) + hash(s.game_id)) % (2**32 - 1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        s.start_time = time.time()
        s.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if s.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        s.cwm = None              # frozen world model
        s.val = None              # online value net
        s.tgt = None              # value target net
        s.opt = None
        s.buf = deque(maxlen=BUF_CAP)

        s.cl = -1                 # current level marker
        s.prev_raw = None
        s.prev_aid = None         # GameAction id 0..7 of last action
        s.prev_cxy = None         # (x,y) of last action if click, else None
        s.prev_levels = 0
        s.step_count = 0
        s.visits = {}             # grid-hash -> visit count (novelty bonus)
        s.eps = 0.5
        s.eps_min = 0.05
        s.eps_decay = 0.9995

    # ---- frame plumbing (mirrors base Agent expectations) ----
    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _levels(s, lf):
        v = getattr(lf, "levels_completed", None)
        return v if v is not None else 0

    def _avail_split(s, avail):
        disc, click = set(), False
        for a in avail or []:
            aid = a.value if hasattr(a, "value") else int(a)
            if 1 <= aid <= 5:
                disc.add(aid)
            elif aid == 6:
                click = True
        return disc, click

    # ---- weight loading ----
    def _load_cwm(s):
        for wp in ("/kaggle/input/forge-pretrained-weights/cwm.pt",
                   os.getenv("ARC_CWM", ""),
                   "cwm.pt"):
            if not wp or not os.path.exists(wp):
                continue
            try:
                state = torch.load(wp, map_location=s.device, weights_only=True)
                ms = s.cwm.state_dict()
                n = 0
                for k, v in state.items():
                    if k in ms and ms[k].shape == v.shape:
                        ms[k] = v; n += 1
                s.cwm.load_state_dict(ms)
                logger.info(f"loaded CWM {n}/{len(ms)} tensors from {wp}")
                return True
            except Exception as e:
                logger.info(f"cwm load failed ({wp}): {e}")
        logger.warning("no CWM weights found -- planner runs on an untrained "
                       "world model (predictions will be random)")
        return False

    def _ensure_nets(s):
        if s.cwm is None:
            s.cwm = ConvWorldModel().to(s.device)
            s._load_cwm()
            s.cwm.eval()
            for p in s.cwm.parameters():
                p.requires_grad_(False)
            s.val = ValueNet().to(s.device)
            s.tgt = deepcopy(s.val).to(s.device)
            for p in s.tgt.parameters():
                p.requires_grad_(False)
            s.opt = optim.Adam(s.val.parameters(), lr=LR)

    def _ext_reward(s, lf):
        # FrameData carries no score field; progress = levels_completed + WIN.
        r = -0.01                                  # mild step cost
        lv = s._levels(lf)
        if lf.state == GameState.WIN:
            r += 10.0
        if lv > s.prev_levels:
            r += 5.0 * (lv - s.prev_levels)
        s.prev_levels = lv
        return r

    # ---- click candidate positions: foreground cells + coarse coverage ----
    def _click_candidates(s, raw):
        cnt = np.bincount(raw.flatten(), minlength=16)
        bg = int(cnt.argmax())
        cands = []
        ys, xs = np.where(raw != bg)
        if len(xs) > 0:
            k = min(len(xs), N_CLICK_CAND * 3 // 4)
            pick = np.random.choice(len(xs), k, replace=False)
            cands.extend((int(xs[i]), int(ys[i])) for i in pick)
        # coarse grid for coverage of empty regions
        rem = N_CLICK_CAND - len(cands)
        if rem > 0:
            step = max(1, G // 6)
            grid_pts = [(x, y) for y in range(step // 2, G, step)
                        for x in range(step // 2, G, step)]
            random.shuffle(grid_pts)
            cands.extend(grid_pts[:rem])
        # dedup, cap
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c); out.append(c)
            if len(out) >= N_CLICK_CAND:
                break
        return out

    # ---- build the candidate action set as parallel arrays ----
    def _candidates(s, raw, disc, click):
        aids, cxys, kinds = [], [], []   # kinds: ("disc",aid) or ("click",x,y)
        for aid in sorted(disc):
            aids.append(aid); cxys.append((-1, -1)); kinds.append(("disc", aid))
        if click:
            for (x, y) in s._click_candidates(raw):
                aids.append(6); cxys.append((x, y)); kinds.append(("click", x, y))
        return aids, cxys, kinds

    # ---- count-based novelty: reward reaching rarely-visited grids ----
    def _novelty(s, grids_long):
        """grids_long [B,64,64] long tensor -> novelty bonus [B] (numpy).
        CURIOSITY / sqrt(1 + visits): unseen grids score highest. The CWM is
        frozen so its prediction error gives no per-game gradient; visit counts
        on actual reached states do, which is what drives exploration before
        any win reward is ever seen."""
        arr = grids_long.detach().cpu().numpy().astype(np.int8)
        out = np.empty(arr.shape[0], dtype=np.float32)
        for i in range(arr.shape[0]):
            v = s.visits.get(arr[i].tobytes(), 0)
            out[i] = CURIOSITY / np.sqrt(1.0 + v)
        return out

    def _visit(s, raw):
        """Record that the agent actually reached this real grid."""
        k = np.asarray(raw, dtype=np.int8).tobytes()
        s.visits[k] = s.visits.get(k, 0) + 1

    @torch.no_grad()
    def _plan(s, raw, disc, click):
        """1-step CWM lookahead. Returns (kind, score_table) for the best action."""
        aids, cxys, kinds = s._candidates(raw, disc, click)
        if not aids:
            return None
        n = len(aids)
        cur = grids_to_long(raw).to(s.device).expand(n, G, G)
        a = torch.tensor(aids, dtype=torch.long, device=s.device)
        cxy = torch.tensor(cxys, dtype=torch.long, device=s.device)
        next_grid, reward, _state_idx, state_logits = s.cwm.imagine(cur, a, cxy)
        win_p = F.softmax(state_logits, dim=-1)[:, WIN_IDX]
        v_next = s.tgt(next_grid)                          # frozen target value
        nov = torch.from_numpy(s._novelty(next_grid)).to(s.device)
        score = reward + WIN_BONUS * win_p + GAMMA * v_next + nov
        best = int(score.argmax().item())
        return kinds[best]

    @torch.no_grad()
    def _plan_bfs(s, raw, disc, click):
        """Beam search over the LEARNED CWM (legitimate model-based planning,
        not source-reading). Returns the FIRST action of the best plan.

        Bounded by design: depth BFS_DEPTH, beam BFS_BEAM. The CWM is ~96%/step
        so error compounds with depth -- shallow + beam-pruned + value at leaves
        keeps the plan from chasing hallucinated deep states. The discounted
        return of a beam node accumulates real predicted reward + P(win) along
        the path; the leaf is bootstrapped with the value net.
        """
        # candidate actions are the same at every node (action set is fixed by
        # what the current frame allows; we re-evaluate availability only at the
        # root, since the CWM cannot tell us the imagined frame's avail set).
        root_aids, root_cxys, root_kinds = s._candidates(raw, disc, click)
        if not root_aids:
            return None

        # beam holds (cumulative_discounted_return, discount_so_far,
        #             grid_long[64,64], first_kind)
        root = grids_to_long(raw).to(s.device)[0]          # [64,64]
        beam = [(0.0, 1.0, root, None, 0.0)]

        for depth in range(BFS_DEPTH):
            # expand every beam node by every candidate action, batched
            grids, firsts, base_ret, base_disc = [], [], [], []
            for (ret, disc_f, g, first, _ns) in beam:
                for k in range(len(root_aids)):
                    grids.append(g)
                    firsts.append(first if first is not None else root_kinds[k])
                    base_ret.append(ret)
                    base_disc.append(disc_f)
            B = len(grids)
            cur = torch.stack(grids, 0)                    # [B,64,64]
            reps = len(beam)
            a = torch.tensor(root_aids * reps, dtype=torch.long, device=s.device)
            cxy = torch.tensor(root_cxys * reps, dtype=torch.long, device=s.device)
            ng, reward, _si, sl = s.cwm.imagine(cur, a, cxy)
            win_p = F.softmax(sl, dim=-1)[:, WIN_IDX]
            v_leaf = s.tgt(ng)
            nov = torch.from_numpy(s._novelty(ng)).to(s.device)
            ret_t = torch.tensor(base_ret, device=s.device)
            dsc_t = torch.tensor(base_disc, device=s.device)
            step_val = reward + WIN_BONUS * win_p + nov
            cum_ret = ret_t + dsc_t * step_val             # return up to here
            # node score = path return + discounted leaf value (for ranking)
            node_score = cum_ret + dsc_t * GAMMA * v_leaf
            # keep top-BFS_BEAM nodes for the next depth
            keep = min(BFS_BEAM, B)
            top = torch.topk(node_score, keep).indices.tolist()
            # carry node_score so the FINAL pick also reflects the leaf value,
            # not just the path return (otherwise the deepest bootstrap is dead)
            beam = [(float(cum_ret[i]), float(dsc_t[i] * GAMMA),
                     ng[i], firsts[i], float(node_score[i])) for i in top]

        # best plan = highest full-estimate beam node; return its FIRST action
        best = int(np.argmax([ns for (_r, _d, _g, _f, ns) in beam]))
        return beam[best][3]

    # ---- value training: real TD + model-based value-iteration backup ----
    def _train(s):
        if len(s.buf) < BATCH:
            return
        batch = [s.buf[i] for i in np.random.choice(len(s.buf), BATCH, replace=False)]
        st = grids_to_long(np.stack([b["s"] for b in batch])).to(s.device)
        st2 = grids_to_long(np.stack([b["s2"] for b in batch])).to(s.device)
        rew = torch.tensor([b["r"] for b in batch], dtype=torch.float32, device=s.device)
        done = torch.tensor([b["d"] for b in batch], dtype=torch.float32, device=s.device)

        v = s.val(st)
        with torch.no_grad():
            v_next = s.tgt(st2)
            td_target = rew + GAMMA * (1 - done) * v_next
        td_loss = F.smooth_l1_loss(v, td_target)

        # model-based value iteration: V(s) <- max_a [ r(s,a) + g V(s') ]
        # over imagined DISCRETE actions (clicks too many to enumerate here).
        with torch.no_grad():
            qs = []
            for aid in range(1, 6):
                a = torch.full((BATCH,), aid, dtype=torch.long, device=s.device)
                ng, r_a, _si, sl = s.cwm.imagine(st, a)
                wp = F.softmax(sl, dim=-1)[:, WIN_IDX]
                qs.append(r_a + WIN_BONUS * wp + GAMMA * s.tgt(ng))
            vi_target = torch.stack(qs, dim=1).max(dim=1).values
        vi_loss = F.smooth_l1_loss(s.val(st), vi_target)

        loss = td_loss + IMAGINE_W * vi_loss
        s.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(s.val.parameters(), 5.0)
        s.opt.step()
        with torch.no_grad():
            for tp, p in zip(s.tgt.parameters(), s.val.parameters()):
                tp.mul_(1 - TAU).add_(TAU * p)

    def is_done(s, frames, lf):
        try:
            return (lf.state == GameState.WIN
                    or (time.time() - s.start_time) >= KAGGLE_RUNTIME_SECONDS - KAGGLE_SAFETY_SECONDS)
        except Exception:
            return True

    def _mk_action(s, kind):
        if kind[0] == "disc":
            sel = GameAction.from_id(kind[1])
            sel.reasoning = f"plan:a{kind[1]}"
            return sel, kind[1], None
        _, x, y = kind
        sel = GameAction.ACTION6
        sel.set_data({"x": int(x), "y": int(y)})
        sel.reasoning = f"plan:c({x},{y})"
        return sel, 6, (int(x), int(y))

    def choose_action(s, frames, lf):
        try:
            if lf.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                s.prev_raw = None; s.prev_aid = None; s.prev_cxy = None
                a = GameAction.RESET; a.reasoning = "reset"
                return a

            s._ensure_nets()
            lvl = s._levels(lf)
            if lvl != s.cl:
                s.cl = lvl
                s.prev_raw = None; s.prev_aid = None; s.prev_cxy = None

            raw = s._raw(lf)
            avail = getattr(lf, "available_actions", None) or []
            disc, click = s._avail_split(avail)

            # only undo / nothing playable
            if not disc and not click:
                ids = {a.value if hasattr(a, "value") else int(a) for a in avail}
                if 7 in ids:
                    a = GameAction.ACTION7; a.reasoning = "only_undo"
                    return a
                a = GameAction.RESET; a.reasoning = "no_actions"
                return a

            s._visit(raw)                       # count this real state for novelty

            # store transition prev -> curr
            if s.prev_raw is not None and s.prev_aid is not None:
                r = s._ext_reward(lf)
                done = 1.0 if lf.state in (GameState.WIN, GameState.GAME_OVER) else 0.0
                s.buf.append({"s": s.prev_raw.copy(), "r": r,
                              "s2": raw.copy(), "d": done})

            s.step_count += 1
            if s.step_count % TRAIN_EVERY == 0:
                s._train()
            s.eps = max(s.eps_min, s.eps * s.eps_decay)

            # epsilon-greedy over the planned best action
            if random.random() < s.eps:
                if click and (not disc or random.random() < 0.5):
                    x, y = random.randint(0, G - 1), random.randint(0, G - 1)
                    kind = ("click", x, y)
                else:
                    kind = ("disc", random.choice(sorted(disc))) if disc \
                        else ("click", random.randint(0, G - 1), random.randint(0, G - 1))
            else:
                kind = (s._plan_bfs(raw, disc, click) if PLANNER == "bfs"
                        else s._plan(raw, disc, click))
                if kind is None:
                    a = GameAction.RESET; a.reasoning = "no_pick"
                    return a

            sel, aid, cxy = s._mk_action(kind)
            s.prev_raw = raw.copy(); s.prev_aid = aid; s.prev_cxy = cxy
            return sel

        except Exception as e:
            traceback.print_exc()
            a = GameAction.from_id(random.choice([1, 2, 3, 4, 5]))
            a.reasoning = f"err:{e}"
            return a
