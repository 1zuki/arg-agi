#!/usr/bin/env python3
"""Compare complete Flash/Sheetu runs by pinned game ID.

This is a read-only diagnostic. It never fills cancelled games or turns an
incomplete offline run into promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: missing or symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected object")
    return value


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _fingerprint(root: Path) -> str:
    """Fingerprint available run identity without relying on timestamps."""
    identity: dict[str, Any] = {}
    for name in ("kernel-metadata.json", "vllm-setup-provenance.json", "taaf_setup_env.json"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_run(root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Load one run and return normalized per-game records."""
    root = Path(root)
    score = _read_json(root, "score.json")
    benchmark = _read_json(root, "benchmark.json")
    games = score.get("games")
    runs = benchmark.get("game_runs")
    if not isinstance(games, dict) or not isinstance(runs, list) or not runs:
        raise ValueError("run is missing score games or benchmark game_runs")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("game_id"), str):
            raise ValueError("benchmark contains an invalid game record")
        game_id = run["game_id"]
        if game_id in seen:
            raise ValueError(f"duplicate game ID: {game_id}")
        seen.add(game_id)
        state = run.get("state")
        if require_complete and state not in {"won", "gave_up"}:
            raise ValueError(f"incomplete game {game_id}: state={state!r}")
        value = run.get("final_score")
        if not _finite(value) or float(value) < 0:
            value = games.get(game_id, {}).get("score") if isinstance(games.get(game_id), dict) else None
        if not _finite(value) or float(value) < 0:
            raise ValueError(f"invalid score for game {game_id}")
        history = run.get("history")
        if not isinstance(history, list):
            raise ValueError(f"invalid history for game {game_id}")
        records.append({
            "game_id": game_id,
            "score": float(value),
            "state": state,
            "actions": len(history),
            "wallclock_seconds": run.get("final_wallclock_seconds"),
        })

    declared_ids = score.get("metadata", {}).get("game_ids")
    if isinstance(declared_ids, list) and declared_ids != [r["game_id"] for r in records]:
        raise ValueError("score metadata game order does not match benchmark order")
    declared_score = score.get("score")
    mean = statistics.fmean(r["score"] for r in records)
    if _finite(declared_score) and not math.isclose(float(declared_score), mean, abs_tol=1e-8):
        raise ValueError("declared score does not match per-game mean")
    return {
        "root": str(root),
        "fingerprint": _fingerprint(root),
        "game_ids": [r["game_id"] for r in records],
        "records": records,
        "mean": mean,
    }


def compare_runs(incumbent: str | Path, candidate: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Compare two runs with identical game IDs and order."""
    base = load_run(incumbent, require_complete=require_complete)
    trial = load_run(candidate, require_complete=require_complete)
    if base["game_ids"] != trial["game_ids"]:
        raise ValueError("runs do not have identical game IDs and order")
    base_by_id = {r["game_id"]: r for r in base["records"]}
    trial_by_id = {r["game_id"]: r for r in trial["records"]}
    deltas = []
    for game_id in base["game_ids"]:
        before = base_by_id[game_id]
        after = trial_by_id[game_id]
        deltas.append({
            "game_id": game_id,
            "incumbent_score": before["score"],
            "candidate_score": after["score"],
            "delta": after["score"] - before["score"],
            "incumbent_actions": before["actions"],
            "candidate_actions": after["actions"],
        })
    values = [d["delta"] for d in deltas]
    return {
        "incumbent": {"root": base["root"], "mean": base["mean"], "fingerprint": base["fingerprint"]},
        "candidate": {"root": trial["root"], "mean": trial["mean"], "fingerprint": trial["fingerprint"]},
        "game_count": len(deltas),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "delta_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "worst_regression": min(values),
        "per_game": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = compare_runs(args.incumbent, args.candidate, require_complete=not args.allow_incomplete)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
