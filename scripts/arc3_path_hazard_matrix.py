#!/usr/bin/env python3
"""Run and aggregate the CPU-only hidden path-hazard policy matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import arc3_generated_game_probe as probe  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "arc3-policy-benchmark-20261101" / "path_hazard"
DEFAULT_GEOMETRIES = tuple(probe.GEOMETRIES)
DEFAULT_SEED_SPLITS = ((100, 199), (200, 299), (1000, 1099))


def _parse_split(value: str) -> tuple[int, int]:
    separator = "-" if "-" in value else ":" if ":" in value else None
    if separator is None:
        raise ValueError(f"seed split must look like START-END, got {value!r}")
    left, right = value.split(separator, 1)
    start, end = int(left), int(right)
    if start > end:
        raise ValueError(f"seed split start must not exceed end: {value!r}")
    return start, end


def _validate_route_invariant(
    geometries: Sequence[str],
    seed_splits: Sequence[tuple[int, int]],
    *,
    levels: int,
    hazard_count: int = 1,
) -> int:
    checked_levels = 0
    for geometry in geometries:
        for start, end in seed_splits:
            for seed in range(start, end + 1):
                game = _make_multi_hazard_game(
                    seed,
                    geometry,
                    levels=levels,
                    hazard_count=hazard_count,
                )
                for level in game.levels:
                    if len(level.path) < hazard_count + 2:
                        raise AssertionError(
                            "path_hazard level lacks enough interior path cells"
                        )
                    if len(level.hazards) != hazard_count:
                        raise AssertionError(
                            f"path_hazard level lacks {hazard_count} hidden hazards"
                        )
                    if not probe._has_public_route(
                        level.start,
                        level.goal,
                        width=level.width,
                        height=level.height,
                        walls=set(level.walls),
                        blocked=level.hazards,
                    ):
                        raise AssertionError(
                            "path_hazard level has no public route around all hazards"
                        )
                    checked_levels += 1
    return checked_levels


def _make_multi_hazard_game(
    seed: int,
    geometry: str,
    *,
    levels: int,
    hazard_count: int,
) -> probe.GeneratedGame:
    """Inject a deterministic hidden-hazard set while preserving public grids."""

    if hazard_count < 1:
        raise ValueError("hazard_count must be a positive integer")
    base = probe.generate_game(seed, "path_hazard", levels=levels, geometry=geometry)
    if hazard_count == 1:
        return base

    updated_levels = []
    for level_index, level in enumerate(base.levels):
        path = list(level.path)
        walls = set(level.walls)
        if len(path) - 2 < hazard_count:
            path = probe._ensure_interior_path(
                level.start,
                level.goal,
                path,
                width=level.width,
                height=level.height,
                rng=random.Random(seed * 1009 + level_index * 7919 + hazard_count),
                minimum_interior=hazard_count,
            )
            # The detour was selected after the base level was generated, so
            # any newly occupied cells must become public free cells.
            walls.difference_update(path)
        candidates = list(path[1:-1])
        if len(candidates) < hazard_count:
            raise RuntimeError(
                f"level {level_index + 1} has only {len(candidates)} interior cells"
            )
        path_cells = set(path)
        while True:
            combinations_list = list(combinations(candidates, hazard_count))
            random.Random(
                seed * 1009 + level_index * 7919 + hazard_count * 104729
            ).shuffle(combinations_list)
            valid = [
                combo
                for combo in combinations_list
                if probe._has_public_route(
                    level.start,
                    level.goal,
                    width=level.width,
                    height=level.height,
                    walls=walls,
                    blocked=frozenset(combo),
                )
            ]
            if valid:
                hazards = frozenset(valid[0])
                break
            removable = sorted(walls - path_cells)
            if not removable:
                raise RuntimeError(
                    "could not preserve an alternate route around all hidden hazards"
                )
            walls.remove(removable[0])
        updated_levels.append(
            replace(
                level,
                path=tuple(path),
                walls=frozenset(walls),
                hazards=hazards,
            )
        )
    return replace(
        base,
        game_id=f"gen-path_hazard-multi{hazard_count}-{geometry}-{seed}",
        levels=tuple(updated_levels),
    )


def _evaluate_games(
    games: Sequence[probe.GeneratedGame],
    policy_type: type[probe.BlindExplorer],
    *,
    max_actions: int,
) -> dict:
    records = [
        probe._run_policy(game, policy_type, max_actions=max_actions) for game in games
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(
            sum(record["won"] for record in records) / max(1, len(records)), 3
        ),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "records": records,
    }


def _policy_stats(records: Sequence[dict]) -> dict[str, int | float]:
    games = len(records)
    stats: dict[str, int | float] = {
        "games": games,
        "wins": sum(int(record["won"]) for record in records),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "actions": sum(record["actions"] for record in records),
        "life_losses": sum(record["life_losses"] for record in records),
        "budget_exhausted": sum(int(record["budget_exhausted"]) for record in records),
    }
    stats["win_rate"] = round(stats["wins"] / games, 6) if games else 0.0
    stats["mean_actions"] = round(stats["actions"] / games, 3) if games else 0.0
    stats["mean_life_losses"] = round(stats["life_losses"] / games, 3) if games else 0.0
    return stats


def _public_policy_stats(stats: dict[str, int | float]) -> dict[str, int | float]:
    return {
        key: stats[key]
        for key in (
            "games",
            "wins",
            "win_rate",
            "levels_completed",
            "levels_total",
            "mean_actions",
            "mean_life_losses",
            "budget_exhausted",
        )
    }


def run_matrix(
    output_dir: Path,
    *,
    geometries: Sequence[str] = DEFAULT_GEOMETRIES,
    seed_splits: Sequence[tuple[int, int]] = DEFAULT_SEED_SPLITS,
    levels: int = 3,
    max_actions: int = 1024,
    hazard_count: int = 1,
) -> dict:
    """Run the matrix, write per-split reports, and return the aggregate."""

    if not geometries:
        raise ValueError("at least one geometry is required")
    if type(levels) is not int or levels < 1:
        raise ValueError("levels must be a positive integer")
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if type(hazard_count) is not int or hazard_count < 1:
        raise ValueError("hazard_count must be a positive integer")
    unknown = [geometry for geometry in geometries if geometry not in probe.GEOMETRIES]
    if unknown:
        raise ValueError(f"unknown geometries: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checked_levels = _validate_route_invariant(
        geometries,
        seed_splits,
        levels=levels,
        hazard_count=hazard_count,
    )
    aggregate: dict[str, dict[str, int | float]] = defaultdict(
        lambda: defaultdict(int)
    )
    reports: list[dict] = []
    source_reports: list[dict] = []

    for geometry in geometries:
        for start, end in seed_splits:
            label = f"{start}_{end}"
            if hazard_count == 1:
                # Keep the established one-hazard report schema and policy
                # ordering unchanged for existing artifacts and comparisons.
                report = probe.evaluate_path_hazard_benchmark(
                    range(start, end + 1),
                    levels=levels,
                    max_actions=max_actions,
                    geometry=geometry,
                )
            else:
                games = [
                    _make_multi_hazard_game(
                        seed,
                        geometry,
                        levels=levels,
                        hazard_count=hazard_count,
                    )
                    for seed in range(start, end + 1)
                ]
                policies = {
                    "baseline": probe.BlindExplorer,
                    "two_phase": probe.TwoPhaseExplorer,
                    "corridor": probe.PublicCorridorExplorer,
                    "adaptive": probe.AdaptiveCorridorExplorer,
                }
                report = {
                    "family": "path_hazard",
                    "geometry": geometry,
                    "action_budget": max_actions,
                    "hazard_count": hazard_count,
                    "policies": {
                        name: _evaluate_games(games, policy, max_actions=max_actions)
                        for name, policy in policies.items()
                    },
                }
            report_path = output_dir / f"{geometry}_{label}.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report_summary = {
                "geometry": geometry,
                "seed_split": label,
                "policies": {},
            }
            expected_games = end - start + 1
            for name, policy in sorted(report["policies"].items()):
                records = policy["records"]
                if len(records) != expected_games:
                    raise AssertionError(
                        f"{geometry}/{label}/{name}: expected {expected_games} games, "
                        f"got {len(records)}"
                    )
                stats = _policy_stats(records)
                report_summary["policies"][name] = _public_policy_stats(stats)
                for key in (
                    "games",
                    "wins",
                    "levels_completed",
                    "levels_total",
                    "actions",
                    "life_losses",
                    "budget_exhausted",
                ):
                    aggregate[name][key] += int(stats[key])
            reports.append(report_summary)
            source_reports.append(
                {
                    "geometry": geometry,
                    "seed_split": label,
                    "path": str(report_path),
                    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                }
            )

    policies = {}
    for name, stats in sorted(aggregate.items()):
        public_stats = _public_policy_stats(_policy_stats([]))
        games = int(stats["games"])
        public_stats.update(
            {
                "games": games,
                "wins": int(stats["wins"]),
                "win_rate": round(int(stats["wins"]) / games, 6) if games else 0.0,
                "levels_completed": int(stats["levels_completed"]),
                "levels_total": int(stats["levels_total"]),
                "mean_actions": round(int(stats["actions"]) / games, 3) if games else 0.0,
                "mean_life_losses": round(int(stats["life_losses"]) / games, 3)
                if games
                else 0.0,
                "budget_exhausted": int(stats["budget_exhausted"]),
            }
        )
        policies[name] = public_stats

    summary = {
        "schema_version": 1,
        "mechanic": "path_hazard" if hazard_count == 1 else "multi_path_hazard",
        "action_budget": max_actions,
        "levels": levels,
        "geometries": list(geometries),
        "seed_splits": [f"{start}-{end}" for start, end in seed_splits],
        "games": sum(end - start + 1 for start, end in seed_splits) * len(geometries),
        "levels_total": checked_levels,
        "route_invariant": {
            "checked_levels": checked_levels,
            "invalid_levels": 0,
            "description": (
                "Every hidden path hazard has at least one public route from start "
                "to goal after the hazard is blocked."
                if hazard_count == 1
                else "The hidden hazard set has at least one public route from start "
                "to goal after all hazards are blocked."
            ),
        },
        "policies": policies,
        "reports": reports,
        "source_reports": source_reports,
        "note": (
            "CPU-only generated-game diagnostic. The scored combined notebook was "
            "not changed and this mechanic has no ARC-AGI-3 transfer evidence."
        ),
    }
    if hazard_count != 1:
        summary["hazard_count"] = hazard_count
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geometries", nargs="+", default=list(DEFAULT_GEOMETRIES))
    parser.add_argument(
        "--seed-splits",
        nargs="+",
        default=[f"{start}-{end}" for start, end in DEFAULT_SEED_SPLITS],
    )
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=1024)
    parser.add_argument("--hazard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        seed_splits = [_parse_split(value) for value in args.seed_splits]
        summary = run_matrix(
            args.output_dir,
            geometries=args.geometries,
            seed_splits=seed_splits,
            levels=args.levels,
            max_actions=args.max_actions,
            hazard_count=args.hazard_count,
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "summary": summary["summary_path"],
                "summary_sha256": summary["summary_sha256"],
                "games": summary["games"],
                "levels_total": summary["levels_total"],
                "policies": summary["policies"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
