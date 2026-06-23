#!/usr/bin/env python3
# =====================================================================
# Compare the two CWM planners head-to-head on the offline engine:
#
#   value : 1-step CWM lookahead  (cheap, shallow)
#   bfs   : depth-N beam search over the learned CWM  (deeper, costlier)
#   mcts  : UCB tree search over the learned CWM       (adaptive, costlier)
#
# Both use the SAME frozen world model and value net; the only difference
# is how far they look ahead before committing to an action. For each game
# we run both and report levels reached / actions used / win, plus average
# planning time per step (the cost side of the trade-off).
#
# This is a LEGITIMATE comparison: bfs searches the *learned* CWM by
# imagining forward, not by reading the hidden game source (that was v31).
#
# Usage:
#   python compare_planners.py --game ls20 ft09 vc33 --max-actions 200
#   python compare_planners.py --game ls20 --bfs-depth 3 --bfs-beam 8 --mcts-sims 32
# =====================================================================
import argparse
import logging
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS_REPO = HERE / "ARC-AGI-3-Agents"
ENV_DIR = HERE / "environment_files"


def _install_light_agents_package():
    agents_dir = AGENTS_REPO / "agents"
    if not agents_dir.is_dir():
        sys.exit(f"agents dir not found at {agents_dir} -- is ARC-AGI-3-Agents cloned here?")
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(agents_dir)]
    sys.modules["agents"] = pkg


def _run_one(gid, planner, max_actions, seed):
    """Run a single game under one planner. Returns a result dict.
    Imports happen inside so ARC_PLANNER (read at module import in my_agent)
    picks up the right value -- we set it before the import, and drop the
    cached module between planners so the constant is re-read."""
    os.environ["ARC_PLANNER"] = planner
    # force a fresh import of my_agent so module-level PLANNER re-reads the env
    for m in ("my_agent",):
        sys.modules.pop(m, None)

    from arc_agi import Arcade, OperationMode
    from my_agent import MyAgent

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    env = arc.make(gid, seed=seed, save_recording=False)
    if env is None:
        return {"game": gid, "planner": planner, "error": "make() returned None"}

    agent = MyAgent(card_id="cmp", game_id=gid, agent_name=f"cmp_{planner}",
                    ROOT_URL="offline", record=False, arc_env=env)
    agent.MAX_ACTIONS = max_actions
    t0 = time.time()
    agent.main()
    dt = time.time() - t0
    lf = agent.frames[-1]
    return {
        "game": gid, "planner": planner,
        "state": lf.state.name,
        "levels": int(lf.levels_completed),
        "actions": int(agent.action_counter),
        "win": lf.state.name == "WIN",
        "sec": round(dt, 1),
        "ms_per_action": round(1000 * dt / max(1, agent.action_counter), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Compare value, bfs, and mcts CWM planners on offline games.")
    ap.add_argument("--game", nargs="+", required=True, help="game id(s), e.g. ls20 ft09")
    ap.add_argument("--max-actions", type=int, default=200)
    ap.add_argument("--planners", nargs="+", default=["value", "bfs", "mcts"],
                    choices=["value", "bfs", "mcts"])
    ap.add_argument("--weights", default="cwm.pt", help="CWM weights (sets ARC_CWM)")
    ap.add_argument("--bfs-depth", type=int, default=3)
    ap.add_argument("--bfs-beam", type=int, default=8)
    ap.add_argument("--mcts-depth", type=int, default=None)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--mcts-cpuct", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
    os.environ["ARC_BFS_DEPTH"] = str(args.bfs_depth)
    os.environ["ARC_BFS_BEAM"] = str(args.bfs_beam)
    os.environ["ARC_MCTS_DEPTH"] = str(args.mcts_depth if args.mcts_depth is not None else args.bfs_depth)
    os.environ["ARC_MCTS_SIMS"] = str(args.mcts_sims)
    os.environ["ARC_MCTS_CPUCT"] = str(args.mcts_cpuct)
    wp = Path(args.weights).resolve()
    if wp.exists():
        os.environ["ARC_CWM"] = str(wp)
        print(f"CWM weights: {wp}")
    else:
        print(f"WARNING: {wp} not found -- planners run on an untrained CWM "
              f"(predictions random; comparison meaningless).", file=sys.stderr)

    sys.path.insert(0, str(HERE))
    _install_light_agents_package()

    # discover availability once
    from arc_agi import Arcade, OperationMode
    arc0 = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    available = {e.game_id.split("-", 1)[0] for e in arc0.get_environments()}

    results = []
    for gid in args.game:
        if gid not in available:
            print(f"!! skip {gid}: not in offline environment_files")
            continue
        for planner in args.planners:
            print(f"running {gid} / {planner} (max {args.max_actions} actions) ...")
            r = _run_one(gid, planner, args.max_actions, args.seed)
            results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  -> {r['state']} levels={r['levels']} actions={r['actions']} "
                      f"win={r['win']} {r['ms_per_action']}ms/action")

    # summary table
    print("\n================ PLANNER COMPARISON ================")
    print(f"{'game':6s} {'planner':7s} {'state':13s} {'lvls':>4s} {'acts':>5s} "
          f"{'win':>4s} {'ms/act':>7s}")
    for r in results:
        if "error" in r:
            print(f"{r['game']:6s} {r['planner']:7s} ERROR: {r['error']}")
            continue
        print(f"{r['game']:6s} {r['planner']:7s} {r['state']:13s} {r['levels']:>4d} "
              f"{r['actions']:>5d} {str(r['win']):>4s} {r['ms_per_action']:>7.1f}")

    # head-to-head per game
    print("\n---- head-to-head (levels reached; higher is better) ----")
    by_game = {}
    for r in results:
        if "error" not in r:
            by_game.setdefault(r["game"], {})[r["planner"]] = r["levels"]
    for g, d in by_game.items():
        parts = [f"{p}={d[p]}" for p in args.planners if p in d]
        if parts:
            best_score = max(d.values())
            winners = [p for p, score in d.items() if score == best_score]
            verdict = "tie:" + ",".join(sorted(winners)) if len(winners) > 1 else winners[0]
            print(f"  {g}: {'  '.join(parts)}  -> {verdict}")


if __name__ == "__main__":
    main()
