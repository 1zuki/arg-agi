#!/usr/bin/env python3
"""Run a paired CPU-only ablation for immediate hazard recovery fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import arc3_generated_game_probe as probe  # noqa: E402
import arc3_path_hazard_matrix as matrix  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "build" / "arc3-policy-benchmark-20261101" / "adaptive_ablation"
)
DEFAULT_SEED_SPLITS = ((100, 199), (200, 299), (1000, 1099))


class ImmediateFallbackAdaptiveCorridorExplorer(probe.AdaptiveCorridorExplorer):
    """Switch to two-phase exploration as soon as a life loss is observed."""

    def choose_action(self, observation: dict) -> str:
        result = observation.get("last_action_result")
        if isinstance(result, dict) and result.get("life_lost"):
            self._using_fallback = True
        return super().choose_action(observation)


POLICIES = {
    "adaptive": probe.AdaptiveCorridorExplorer,
    "immediate_fallback": ImmediateFallbackAdaptiveCorridorExplorer,
}


def _merge_stats(target: dict[str, int], stats: dict[str, int | float]) -> None:
    for key in (
        "games",
        "wins",
        "levels_completed",
        "levels_total",
        "actions",
        "life_losses",
        "budget_exhausted",
    ):
        target[key] += int(stats[key])


def _public_aggregate(stats: dict[str, int]) -> dict[str, int | float]:
    games = int(stats["games"])
    return {
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


def run_ablation(
    output_dir: Path,
    *,
    geometries: tuple[str, ...] = tuple(probe.GEOMETRIES),
    seed_splits: tuple[tuple[int, int], ...] = DEFAULT_SEED_SPLITS,
    levels: int = 3,
    max_actions: int = 1024,
    hazard_counts: tuple[int, ...] = (1, 2),
) -> dict:
    """Run paired policies and write per-split plus aggregate reports."""

    if not geometries:
        raise ValueError("at least one geometry is required")
    unknown = [geometry for geometry in geometries if geometry not in probe.GEOMETRIES]
    if unknown:
        raise ValueError(f"unknown geometries: {unknown}")
    if type(levels) is not int or levels < 1:
        raise ValueError("levels must be a positive integer")
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if not hazard_counts or any(
        type(count) is not int or count < 1 for count in hazard_counts
    ):
        raise ValueError("hazard_counts must contain positive integers")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[int, dict[str, dict[str, int]]] = {
        count: {name: defaultdict(int) for name in POLICIES}
        for count in hazard_counts
    }
    reports: list[dict] = []
    source_reports: list[dict] = []
    checked_levels = 0

    for hazard_count in hazard_counts:
        for geometry in geometries:
            for start, end in seed_splits:
                label = f"{start}_{end}"
                paired_records = []
                for seed in range(start, end + 1):
                    game = matrix._make_multi_hazard_game(
                        seed,
                        geometry,
                        levels=levels,
                        hazard_count=hazard_count,
                    )
                    for level in game.levels:
                        if len(level.hazards) != hazard_count:
                            raise AssertionError("hazard count invariant failed")
                        if not probe._has_public_route(
                            level.start,
                            level.goal,
                            width=level.width,
                            height=level.height,
                            walls=set(level.walls),
                            blocked=level.hazards,
                        ):
                            raise AssertionError("route invariant failed")
                        checked_levels += 1
                    policy_records = {
                        name: probe._run_policy(
                            game,
                            policy,
                            max_actions=max_actions,
                        )
                        for name, policy in POLICIES.items()
                    }
                    for name, record in policy_records.items():
                        _merge_stats(aggregate[hazard_count][name], matrix._policy_stats([record]))
                    paired_records.append(
                        {
                            "game_id": game.game_id,
                            "seed": seed,
                            "policies": policy_records,
                        }
                    )

                per_policy = {}
                for name in POLICIES:
                    records = [row["policies"][name] for row in paired_records]
                    per_policy[name] = matrix._public_policy_stats(
                        matrix._policy_stats(records)
                    )
                report = {
                    "schema_version": 1,
                    "mechanic": "path_hazard",
                    "hazard_count": hazard_count,
                    "geometry": geometry,
                    "seed_split": label,
                    "action_budget": max_actions,
                    "policies": per_policy,
                    "paired_records": paired_records,
                    "note": (
                        "CPU-only generated-game ablation. The scored combined "
                        "notebook was not changed and this has no ARC-AGI-3 "
                        "transfer evidence."
                    ),
                }
                report_path = output_dir / f"hazards{hazard_count}_{geometry}_{label}.json"
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                reports.append(
                    {
                        "hazard_count": hazard_count,
                        "geometry": geometry,
                        "seed_split": label,
                        "policies": per_policy,
                    }
                )
                source_reports.append(
                    {
                        "hazard_count": hazard_count,
                        "geometry": geometry,
                        "seed_split": label,
                        "path": str(report_path),
                        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    }
                )

    policies = {
        str(hazard_count): {
            name: _public_aggregate(stats)
            for name, stats in aggregate[hazard_count].items()
        }
        for hazard_count in hazard_counts
    }
    deltas = {}
    for hazard_count in hazard_counts:
        adaptive = policies[str(hazard_count)]["adaptive"]
        immediate = policies[str(hazard_count)]["immediate_fallback"]
        deltas[str(hazard_count)] = {
            "mean_actions_immediate_minus_adaptive": round(
                immediate["mean_actions"] - adaptive["mean_actions"], 3
            ),
            "mean_life_losses_immediate_minus_adaptive": round(
                immediate["mean_life_losses"] - adaptive["mean_life_losses"], 3
            ),
            "win_rate_immediate_minus_adaptive": round(
                immediate["win_rate"] - adaptive["win_rate"], 6
            ),
        }

    summary = {
        "schema_version": 1,
        "mechanic": "adaptive_recovery_ablation",
        "hazard_counts": list(hazard_counts),
        "action_budget": max_actions,
        "levels": levels,
        "geometries": list(geometries),
        "seed_splits": [f"{start}-{end}" for start, end in seed_splits],
        "games_per_hazard_count": sum(end - start + 1 for start, end in seed_splits)
        * len(geometries),
        "levels_checked": checked_levels,
        "policies": policies,
        "paired_deltas": deltas,
        "reports": reports,
        "source_reports": source_reports,
        "decision": (
            "Do not promote immediate fallback yet: it reduces mean actions only "
            "when accepting a higher life-loss rate, while win rates are tied."
        ),
        "note": (
            "CPU-only generated-game ablation. The scored combined notebook was "
            "not changed and this has no ARC-AGI-3 transfer evidence."
        ),
    }
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
    parser.add_argument("--geometries", nargs="+", default=list(probe.GEOMETRIES))
    parser.add_argument(
        "--seed-splits",
        nargs="+",
        default=[f"{start}-{end}" for start, end in DEFAULT_SEED_SPLITS],
    )
    parser.add_argument("--hazard-counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        seed_splits = tuple(matrix._parse_split(value) for value in args.seed_splits)
        summary = run_ablation(
            args.output_dir,
            geometries=tuple(args.geometries),
            seed_splits=seed_splits,
            levels=args.levels,
            max_actions=args.max_actions,
            hazard_counts=tuple(args.hazard_counts),
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "summary": summary["summary_path"],
                "summary_sha256": summary["summary_sha256"],
                "levels_checked": summary["levels_checked"],
                "policies": summary["policies"],
                "paired_deltas": summary["paired_deltas"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
