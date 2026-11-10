"""Run a bounded CPU-only compatibility smoke against local ARC-AGI-3 games.

This checks environment discovery, reset, action schemas, and one executable
action per selected game. It deliberately does not load a language model or
claim gameplay quality, solver quality, or a competition score.

The official ``arc-agi`` runtime is supplied by the competition wheelhouse;
install it in an isolated Python 3.12 environment before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _probe_action(frame: Any, action_id: int) -> tuple[Any, dict[str, int]]:
    """Build a schema-valid action for the current frame.

    ``ACTION6`` is the complex click action and requires display coordinates;
    the frame centre is only a compatibility probe, not a gameplay heuristic.
    """

    from arcengine import GameAction

    action = GameAction.from_id(action_id)
    if action is not GameAction.ACTION6:
        return action, {}
    grid = frame.frame[0]
    height, width = grid.shape
    return action, {"x": int(width // 2), "y": int(height // 2)}


def run_smoke(
    environment_dir: Path,
    *,
    max_games: int | None = None,
    actions_per_game: int = 1,
) -> dict[str, Any]:
    """Reset and probe each selected local game, returning JSON-safe results."""

    if actions_per_game < 1:
        raise ValueError("actions_per_game must be at least 1")

    import arc_agi

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environment_dir),
    )
    discovered = list(arcade.available_environments)
    selected = discovered if max_games is None else discovered[:max_games]
    records: list[dict[str, Any]] = []

    for info in selected:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "game_id": info.game_id,
            "reset_ok": False,
            "actions_ok": 0,
            "available_actions": [],
            "state": None,
            "error": None,
        }
        try:
            environment = arcade.make(
                info.game_id,
                save_recording=False,
                include_frame_data=True,
            )
            if environment is None:
                raise RuntimeError("arcade.make returned None")
            frame = environment.reset()
            if frame is None:
                raise RuntimeError("reset returned None")
            record["reset_ok"] = True
            record["available_actions"] = list(frame.available_actions or [])
            record["state"] = getattr(frame.state, "value", str(frame.state))

            for _ in range(actions_per_game):
                available = list(frame.available_actions or [])
                if not available:
                    raise RuntimeError("frame exposed no available actions")
                action, data = _probe_action(frame, int(available[0]))
                frame = environment.step(action, data=data)
                if frame is None:
                    raise RuntimeError(f"action {action.name} returned None")
                record["actions_ok"] += 1
                record["state"] = getattr(frame.state, "value", str(frame.state))
        except Exception as exc:  # report every game so one failure is diagnosable
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_s"] = round(time.perf_counter() - started, 4)
        records.append(record)

    return {
        "environment_dir": str(environment_dir),
        "discovered_games": len(discovered),
        "selected_games": len(selected),
        "reset_ok": sum(bool(item["reset_ok"]) for item in records),
        "actions_ok": sum(int(item["actions_ok"]) for item in records),
        "errors": [item for item in records if item["error"]],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-dir", type=Path, default=Path("environment_files"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--actions-per-game", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_smoke(
        args.environment_dir,
        max_games=args.max_games,
        actions_per_game=args.actions_per_game,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")

    expected_actions = result["selected_games"] * args.actions_per_game
    return int(
        bool(
            result["errors"]
            or result["reset_ok"] != result["selected_games"]
            or result["actions_ok"] != expected_actions
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
