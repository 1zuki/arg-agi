"""Replay and audit a local Skyfall ARC-AGI-3 game shelf.

This is a CPU-only compatibility and provenance smoke.  It consumes already
packaged games; it does not invoke the generator, an agent CLI, an API, a
language model, or a competition submission.  A passing result proves that the
packaged trajectories and their authored layouts are still executable under
the installed ``arcengine`` runtime.  It says nothing about hidden-game
generalization or an ARC-AGI-3 leaderboard score.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


TSV_COLUMNS = (
    "step",
    "observation",
    "info",
    "level",
    "action_valid",
    "duration",
    "action",
    "reward",
    "terminated",
    "truncated",
    "explore_exploit",
    "world_model_path",
    "goal",
    "exploration_output",
)
VERBS = {
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    "INTERACT": 5,
    "CLICK": 6,
    "ACTION7": 7,
}


class ShelfSmokeError(RuntimeError):
    """A packaged-game contract or replay failure."""


def _load_game_class(game_path: Path):
    from arcengine import ARCBaseGame

    module_name = "arc3_shelf_" + game_path.parent.parent.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, game_path)
    if spec is None or spec.loader is None:
        raise ShelfSmokeError(f"could not load {game_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ARCBaseGame)
        and value is not ARCBaseGame
        and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ShelfSmokeError(
            f"{game_path}: expected one game class defined in the module, found {len(classes)}"
        )
    return classes[0]


def _grid_is_valid(value: str, *, label: str) -> None:
    try:
        grid = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ShelfSmokeError(f"{label}: observation is not JSON: {exc}") from None
    if not isinstance(grid, list) or len(grid) != 64:
        raise ShelfSmokeError(f"{label}: expected a 64-row observation")
    for row in grid:
        if not isinstance(row, list) or len(row) != 64:
            raise ShelfSmokeError(f"{label}: expected a 64x64 observation")
        if any(not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 15 for cell in row):
            raise ShelfSmokeError(f"{label}: observation contains a non-palette cell")


def _parse_action(value: str, *, label: str) -> tuple[int, dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ShelfSmokeError(f"{label}: action is not JSON: {exc}") from None
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ShelfSmokeError(f"{label}: action must be [verb, payload]")
    verb, payload = parsed
    if verb not in VERBS or not isinstance(payload, dict):
        raise ShelfSmokeError(f"{label}: unsupported action {parsed!r}")
    if verb == "CLICK":
        if set(payload) != {"x", "y"} or any(
            isinstance(payload[key], bool) or not isinstance(payload[key], int)
            for key in ("x", "y")
        ):
            raise ShelfSmokeError(f"{label}: CLICK payload must contain integer x and y")
    elif payload:
        raise ShelfSmokeError(f"{label}: non-click action payload must be empty")
    return VERBS[verb], payload


def _read_rows(tsv_path: Path, slug: str) -> list[dict[str, str]]:
    try:
        with tsv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
                raise ShelfSmokeError(f"{slug}: packaged TSV header does not match the contract")
            rows = list(reader)
    except OSError as exc:
        raise ShelfSmokeError(f"{slug}: cannot read {tsv_path}: {exc}") from None
    if not rows:
        raise ShelfSmokeError(f"{slug}: packaged TSV is empty")
    if rows[-1]["action"] != "":
        raise ShelfSmokeError(f"{slug}: packaged TSV has no blank terminal row")
    for index, row in enumerate(rows, start=1):
        _grid_is_valid(row["observation"], label=f"{slug}: row {index}")
        try:
            json.loads(row["info"])
            int(row["step"])
            int(row["level"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ShelfSmokeError(f"{slug}: malformed metadata at row {index}: {exc}") from None
        if row["action"]:
            _parse_action(row["action"], label=f"{slug}: row {index}")
    return rows


def _manifest_and_metadata(game_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = json.loads((game_dir / "package" / "manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((game_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShelfSmokeError(f"{game_dir}: invalid manifest or metadata: {exc}") from None
    if manifest.get("verdict") != "packaged":
        raise ShelfSmokeError(f"{game_dir.name}: manifest is not packaged")
    baseline = metadata.get("baseline_actions")
    if not isinstance(baseline, list) or not baseline:
        raise ShelfSmokeError(f"{game_dir.name}: metadata has no baseline action list")
    seeds = manifest.get("seeds") or {}
    validated = seeds.get("validated")
    authored_seed = seeds.get("authored_seed")
    if seeds.get("requested") != 5 or not isinstance(validated, list) or len(validated) != 5:
        raise ShelfSmokeError(f"{game_dir.name}: expected five requested and validated seeds")
    if isinstance(authored_seed, bool) or not isinstance(authored_seed, int):
        raise ShelfSmokeError(f"{game_dir.name}: invalid authored seed")
    for entry in validated:
        if (
            not isinstance(entry, dict)
            or entry.get("win") is not True
            or entry.get("levels_completed") != len(baseline)
            or entry.get("error") is not None
        ):
            raise ShelfSmokeError(f"{game_dir.name}: a validated seed does not claim a clean win")
    authored = [entry for entry in validated if entry.get("seed") == authored_seed]
    if len(authored) != 1:
        raise ShelfSmokeError(f"{game_dir.name}: authored seed is missing from validation results")
    return manifest, metadata


def _audit_seed_layouts(module: Any, manifest: dict[str, Any], slug: str) -> int:
    """Re-derive every validated layout and check determinism/distinctness."""

    build_layout = getattr(module, "build_layout", None)
    if not callable(build_layout):
        raise ShelfSmokeError(f"{slug}: manifest seeds exist but build_layout is not callable")
    entries = manifest["seeds"]["validated"]
    serialized: list[str] = []
    for entry in entries:
        seed = entry["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ShelfSmokeError(f"{slug}: invalid validated layout seed {seed!r}")
        try:
            first = json.dumps(build_layout(seed), sort_keys=True, separators=(",", ":"))
            second = json.dumps(build_layout(seed), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ShelfSmokeError(f"{slug}: build_layout({seed}) is not JSON serializable: {exc}") from None
        if first != second:
            raise ShelfSmokeError(f"{slug}: build_layout({seed}) is not deterministic")
        serialized.append(first)
    if len(serialized) != len(set(serialized)):
        raise ShelfSmokeError(f"{slug}: distinct validated seeds produced duplicate layouts")
    return len(serialized)


def audit_game(game_dir: Path) -> dict[str, Any]:
    """Audit and replay one packaged shelf entry."""

    slug = game_dir.parent.name
    manifest, metadata = _manifest_and_metadata(game_dir)
    rows = _read_rows(game_dir / "package" / "tsv" / "level-set.tsv", slug)
    game_cls = _load_game_class(game_dir / "game.py")
    module_path = game_dir / "game.py"
    module_name = "arc3_shelf_" + slug.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ShelfSmokeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validated_seed_layouts = _audit_seed_layouts(module, manifest, slug)
    seed = int(manifest["seeds"]["authored_seed"])
    os.environ.pop("ONLY_RESET_LEVELS", None)
    game = game_cls(layout_seed=seed)
    from arcengine import ActionInput, GameAction

    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    action_rows = [row for row in rows if row["action"]]
    authored_entry = next(entry for entry in manifest["seeds"]["validated"] if entry["seed"] == seed)
    expected_levels = len(metadata["baseline_actions"])
    for index, row in enumerate(action_rows, start=1):
        action_id, payload = _parse_action(row["action"], label=f"{slug}: row {index}")
        expected_level = int(frame.levels_completed) + 1
        if int(row["level"]) != expected_level:
            raise ShelfSmokeError(f"{slug}: row {index} level is {row['level']}, expected {expected_level}")
        available = {int(value) for value in (frame.available_actions or [])}
        if action_id not in available:
            raise ShelfSmokeError(f"{slug}: row {index} action {action_id} is not available")
        frame = game.perform_action(
            ActionInput(id=getattr(GameAction, f"ACTION{action_id}"), data=payload)
        )
        if frame.state.name in {"WIN", "GAME_OVER"} and index != len(action_rows):
            raise ShelfSmokeError(f"{slug}: terminal state appeared before the trajectory ended")
    terminal = rows[-1]
    if frame.state.name != "WIN" or int(frame.levels_completed) != expected_levels:
        raise ShelfSmokeError(
            f"{slug}: replay ended at {frame.state.name}, levels={frame.levels_completed}/{expected_levels}"
        )
    if int(authored_entry["actions_total"]) != len(action_rows):
        raise ShelfSmokeError(f"{slug}: manifest action count does not match packaged trajectory")
    if terminal["terminated"] != "True" or terminal["action"] != "":
        raise ShelfSmokeError(f"{slug}: terminal row flags are invalid")
    return {
        "slug": slug,
        "seed": seed,
        "levels": expected_levels,
        "actions": len(action_rows),
        "validated_seed_layouts": validated_seed_layouts,
        "manifest_seed_win_claims": len(manifest["seeds"]["validated"]),
        "state": frame.state.name,
    }


def audit_shelf(shelf_root: Path, *, max_games: int | None = None) -> dict[str, Any]:
    """Audit packaged games under ``shelf_root/games``."""

    game_dirs = sorted(path.parent.parent for path in (shelf_root / "games").glob("*/game/package/manifest.json"))
    if max_games is not None:
        if max_games < 1:
            raise ValueError("max_games must be at least 1")
        game_dirs = game_dirs[:max_games]
    if not game_dirs:
        raise ShelfSmokeError(f"no packaged games found under {shelf_root / 'games'}")
    results = [audit_game(path) for path in game_dirs]
    return {
        "shelf_root": str(shelf_root),
        "games": len(results),
        "canonical_replays_won": sum(item["state"] == "WIN" for item in results),
        "validated_seed_layouts": sum(item["validated_seed_layouts"] for item in results),
        "manifest_seed_win_claims": sum(item["manifest_seed_win_claims"] for item in results),
        "total_levels": sum(item["levels"] for item in results),
        "total_actions": sum(item["actions"] for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shelf-root",
        type=Path,
        default=Path("/tmp/arc-agi-3-games-research"),
        help="Skyfall shelf checkout containing games/ (default: %(default)s)",
    )
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_shelf(args.shelf_root, max_games=args.max_games)
    except (ImportError, OSError, ShelfSmokeError, ValueError, TypeError) as exc:
        print(f"arc3_skyfall_shelf_smoke: FAIL: {exc}")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
