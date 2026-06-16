#!/usr/bin/env python3
# =====================================================================
# Local runner for the Dyna-WM agent against the OFFLINE ARC-AGI engine.
#
# Why this exists instead of `uv run main.py`:
#   - main.py discovers games via HTTP GET {ROOT_URL}/api/games, so it
#     needs the Flask gateway running even in "offline" play.
#   - agents/__init__.py imports langgraph / smolagents / langsmith,
#     none of which are installed (or needed) for our agent.
#   This runner skips both: it builds a lightweight `agents` package
#   (empty __init__) so `from agents.agent import Agent` works without
#   the heavy templates, then drives Agent.main() on a LocalEnvironmentWrapper.
#
# Recordings land in RECORDINGS_DIR (default ./recordings) as
#   {game_id}.myagent.{guid}.recording.jsonl  -> feed straight to replay.py
#
# Usage:
#   python run_local.py --game ls20 --max-actions 200
#   python run_local.py --game ls20 --weights pretrained_weights.pt
#   python run_local.py --game ls20 ft09 vc33 --max-actions 500
# =====================================================================
import argparse
import logging
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS_REPO = HERE / "ARC-AGI-3-Agents"
ENV_DIR = HERE / "environment_files"


def _install_light_agents_package():
    """Make `from agents.agent import Agent` importable WITHOUT running the
    real agents/__init__.py (which pulls in langgraph/smolagents/langsmith)."""
    agents_dir = AGENTS_REPO / "agents"
    if not agents_dir.is_dir():
        sys.exit(f"agents dir not found at {agents_dir} — is ARC-AGI-3-Agents cloned here?")
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(agents_dir)]          # so submodules resolve
    sys.modules["agents"] = pkg               # pre-empt the heavy __init__.py


def main():
    ap = argparse.ArgumentParser(description="Run the Dyna-WM agent on offline ARC-AGI-3 games.")
    ap.add_argument("--game", nargs="+", required=True, help="game id(s), e.g. ls20 ft09")
    ap.add_argument("--max-actions", type=int, default=200,
                    help="cap actions per game so a non-winning game doesn't run for hours")
    ap.add_argument("--weights", default=None,
                    help="pretrained weights path (sets ARC_PRETRAINED so the agent loads it)")
    ap.add_argument("--recordings-dir", default=str(HERE / "recordings"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    os.environ["RECORDINGS_DIR"] = args.recordings_dir
    os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
    if args.weights:
        wp = Path(args.weights).resolve()
        if not wp.exists():
            sys.exit(f"weights not found: {wp}")
        os.environ["ARC_PRETRAINED"] = str(wp)
        print(f"using pretrained weights: {wp}")
    else:
        print("no --weights given: agent learns from scratch this run")

    # agent source is in HERE; the light agents package points at the repo
    sys.path.insert(0, str(HERE))
    _install_light_agents_package()

    from arc_agi import Arcade, OperationMode
    from arcengine import GameState
    from my_agent import MyAgent

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    available = {e.game_id.split("-", 1)[0] for e in arc.get_environments()}
    print(f"offline games available: {sorted(available)}")

    for gid in args.game:
        if gid not in available:
            print(f"  !! skip {gid}: not in offline environment_files")
            continue
        print(f"\n===== playing {gid} (max {args.max_actions} actions) =====")
        env = arc.make(gid, seed=args.seed, save_recording=False)
        if env is None:
            print(f"  make() returned None for {gid}")
            continue
        agent = MyAgent(
            card_id="local",
            game_id=gid,
            agent_name="myagent",
            ROOT_URL="offline",
            record=True,                       # Agent's own Recorder -> RECORDINGS_DIR
            arc_env=env,
        )
        agent.MAX_ACTIONS = args.max_actions   # bound the run locally
        agent.main()                           # base-class loop: choose_action -> step -> record
        lf = agent.frames[-1]
        print(f"  result: state={lf.state.name} levels_completed={lf.levels_completed} "
              f"actions={agent.action_counter} fps={agent.fps}")
        if hasattr(agent, "recorder"):
            print(f"  recording: {agent.recorder.filename}")

    print(f"\nreplay with:\n  python replay.py {args.recordings_dir}")


if __name__ == "__main__":
    main()
