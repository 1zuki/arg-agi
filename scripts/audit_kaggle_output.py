#!/usr/bin/env python3
"""Fail-closed audit of downloaded outputs for the ARC-AGI-3 candidate."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CANDIDATE_VARIANT = "taaf-qwen38-xhigh-checkpoint8"
MODEL_DATASET_REF = "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
MODEL_HF_REF = "Qwen/Qwen3.8-27B-FP8"
MODEL_HF_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MODEL_REASONING_EFFORT = "xhigh"
EXPECTED_DATASET_VERSIONS = {
    "jeroencottaar/taaf-kaggle-source-share": 4,
    "driessmit1/arc3-vllm-h100-wheelhouse-v3": 1,
    MODEL_DATASET_REF: 1,
}
EXPECTED_SOURCE_TREE = {
    "sha256": "f8554ee8224e0701266459a9322ce4553e94f9239539196fee27992db4c7e11f",
    "file_count": 73,
}
EXPECTED_PATCHED_TREE = {
    "sha256": "686d07fe0d69cd73942f14de4c4f0c165b1956009d1eb9bdd439dd4b7b9e5c30",
    "file_count": 73,
}
EXPECTED_SOLVER_DEFAULTS = {
    "concurrency": 28,
    "analyzer_timeout": 900.0,
    "max_runtime_s_per_game": 7920.0,
}
EXPECTED_DEPLOYMENT_MAX_RUNTIME_S = 32400.0
EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S = 1800.0
EXPECTED_MODEL_CRC32_CORRECTION_COUNT = 3
SOLVER_OVERRIDE_ENV = {
    "concurrency": "TAAF_BENCHMARK_CONCURRENCY",
    "analyzer_timeout": "TAAF_ANALYZER_TIMEOUT",
    "max_runtime_s_per_game": "TAAF_MAX_RUNTIME_S_PER_GAME",
}
EXPECTED_PUBLIC_GAME_COUNT = 25
EXPECTED_PUBLIC_GAME_IDS = {
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
}
EXPECTED_N_PASSES = 1
MAX_SELECTED_GAME_COUNT = 1000
COMPLETED_GAME_STATES = {"won", "gave_up"}
FATAL_LOG_PATTERNS = (
    (
        "python_traceback",
        re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    ),
    (
        "cuda_oom",
        re.compile(
            r"(?:cuda out of memory|outofmemoryerror|memoryerror:|oom-kill)",
            re.IGNORECASE,
        ),
    ),
    ("fatal_python", re.compile(r"fatal python error", re.IGNORECASE)),
    ("segfault", re.compile(r"(?:segmentation fault|segfault)", re.IGNORECASE)),
    (
        "engine_startup",
        re.compile(r"(?:enginecore|engine core).*(?:failed|error)", re.IGNORECASE),
    ),
    (
        "server_startup",
        re.compile(r"failed to (?:start|initialize).*(?:server|engine)", re.IGNORECASE),
    ),
)
MAX_LOG_MATCHES_PER_PATTERN = 5
MAX_JSON_BYTES = {
    "taaf_run_manifest.json": 4 * 1024 * 1024,
    "benchmark.json": 128 * 1024 * 1024,
    "taaf_submission_health.json": 64 * 1024,
}
MAX_VLLM_LOG_BYTES = 512 * 1024 * 1024


class Audit:
    def __init__(
        self, output_dir: Path, *, mode: str, allow_solver_overrides: bool
    ) -> None:
        self.output_dir = output_dir
        self.mode = mode
        self.allow_solver_overrides = allow_solver_overrides
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.summary: dict[str, Any] = {}

    def check(self, name: str, passed: bool, detail: str) -> bool:
        if len(detail) > 2000:
            detail = detail[:1997] + "..."
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")
        return bool(passed)

    def warn(self, message: str) -> None:
        if len(message) > 2000:
            message = message[:1997] + "..."
        self.warnings.append(message)

    def report(self) -> dict[str, Any]:
        return {
            "audit_version": 1,
            "candidate": CANDIDATE_VARIANT,
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "passed": not self.errors,
            "summary": self.summary,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _same_number(actual: Any, expected: float) -> bool:
    return _is_finite_number(actual) and float(actual) == float(expected)


def _same_env_number(actual: Any, expected: float) -> bool:
    if (
        not isinstance(actual, str)
        or not actual.strip()
        or not _is_finite_number(expected)
    ):
        return False
    try:
        parsed = float(actual)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed == float(expected)


def _is_regular_nonsymlink(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _read_json_object(audit: Audit, filename: str) -> dict[str, Any] | None:
    path = audit.output_dir / filename
    if not audit.check(
        f"{filename}.exists",
        _is_regular_nonsymlink(path),
        f"expected non-symlink regular file {path}",
    ):
        return None
    size = path.stat().st_size
    max_bytes = MAX_JSON_BYTES[filename]
    if not audit.check(
        f"{filename}.size",
        0 < size <= max_bytes,
        f"expected 1..{max_bytes} bytes, got {size}",
    ):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        audit.check(
            f"{filename}.json",
            False,
            f"could not parse JSON: {type(exc).__name__}: {exc}",
        )
        return None
    if not audit.check(
        f"{filename}.object",
        isinstance(value, dict),
        "top-level JSON must be an object",
    ):
        return None
    audit.check(f"{filename}.json", True, "valid JSON object")
    return value


def _check_equal(audit: Audit, name: str, actual: Any, expected: Any) -> None:
    audit.check(name, actual == expected, f"expected {expected!r}, got {actual!r}")


def _audit_manifest(audit: Audit) -> dict[str, Any] | None:
    manifest = _read_json_object(audit, "taaf_run_manifest.json")
    if manifest is None:
        return None

    _check_equal(
        audit, "manifest.variant", manifest.get("submission_variant"), CANDIDATE_VARIANT
    )
    _check_equal(
        audit, "manifest.milestone", manifest.get("milestone"), "benchmark_completed"
    )
    _check_equal(audit, "manifest.outcome", manifest.get("outcome"), "completed")
    audit.check(
        "manifest.error_absent",
        "error" not in manifest,
        "completed manifest must not include an error",
    )
    _check_equal(
        audit,
        "manifest.dataset_versions",
        manifest.get("expected_dataset_versions"),
        EXPECTED_DATASET_VERSIONS,
    )
    _check_equal(
        audit,
        "manifest.batch_checkpoint_limit",
        manifest.get("batch_checkpoint_limit"),
        8,
    )

    true_submission = manifest.get("TRUE_SUBMISSION")
    expected_submission = audit.mode == "submission"
    _check_equal(
        audit, "manifest.submission_mode", true_submission, expected_submission
    )

    model = manifest.get("model")
    if audit.check(
        "manifest.model.object",
        isinstance(model, dict),
        f"expected object, got {type(model).__name__}",
    ):
        _check_equal(
            audit,
            "manifest.model.dataset",
            model.get("kaggle_dataset_ref"),
            MODEL_DATASET_REF,
        )
        _check_equal(
            audit, "manifest.model.hf_ref", model.get("huggingface_ref"), MODEL_HF_REF
        )
        _check_equal(
            audit,
            "manifest.model.revision",
            model.get("huggingface_revision"),
            MODEL_HF_REVISION,
        )
        _check_equal(
            audit, "manifest.model.served_name", model.get("served_name"), MODEL_HF_REF
        )
        _check_equal(
            audit,
            "manifest.model.reasoning",
            model.get("reasoning_effort"),
            MODEL_REASONING_EFFORT,
        )

    bundle_patch = manifest.get("bundle_patch")
    if audit.check(
        "manifest.bundle_patch.object",
        isinstance(bundle_patch, dict),
        f"expected object, got {type(bundle_patch).__name__}",
    ):
        audit.check(
            "manifest.bundle_patch.status",
            bundle_patch.get("status")
            in {"verified_and_applied", "verified_existing_copy"},
            f"unexpected status {bundle_patch.get('status')!r}",
        )
        _check_equal(
            audit,
            "manifest.bundle_patch.source_tree",
            bundle_patch.get("source_tree"),
            EXPECTED_SOURCE_TREE,
        )
        _check_equal(
            audit,
            "manifest.bundle_patch.patched_tree",
            bundle_patch.get("patched_tree"),
            EXPECTED_PATCHED_TREE,
        )

    wheelhouse = manifest.get("wheelhouse")
    if audit.check(
        "manifest.wheelhouse.object",
        isinstance(wheelhouse, dict),
        f"expected object, got {type(wheelhouse).__name__}",
    ):
        _check_equal(
            audit,
            "manifest.wheelhouse.status",
            wheelhouse.get("status"),
            "inventory_and_sha256_verified",
        )
        _check_equal(
            audit, "manifest.wheelhouse.file_count", wheelhouse.get("file_count"), 180
        )
        _check_equal(
            audit,
            "manifest.wheelhouse.payload_count",
            wheelhouse.get("payload_sha256_count"),
            178,
        )
        _check_equal(
            audit,
            "manifest.wheelhouse.versions",
            wheelhouse.get("core_versions"),
            {"vllm": "0.19.0", "torch": "2.10.0", "transformers": "4.57.6"},
        )

    model_snapshot = manifest.get("model_snapshot")
    if audit.check(
        "manifest.model_snapshot.object",
        isinstance(model_snapshot, dict),
        f"expected object, got {type(model_snapshot).__name__}",
    ):
        _check_equal(
            audit,
            "manifest.model_snapshot.status",
            model_snapshot.get("status"),
            "inventory_and_crc32_verified",
        )
        _check_equal(
            audit,
            "manifest.model_snapshot.file_count",
            model_snapshot.get("file_count"),
            80,
        )
        _check_equal(
            audit,
            "manifest.model_snapshot.crc_count",
            model_snapshot.get("crc32_count"),
            77,
        )
        _check_equal(
            audit,
            "manifest.model_snapshot.crc_correction_count",
            model_snapshot.get("crc32_correction_count"),
            EXPECTED_MODEL_CRC32_CORRECTION_COUNT,
        )
        _check_equal(
            audit,
            "manifest.model_snapshot.shards",
            model_snapshot.get("weight_shard_count"),
            66,
        )
        _check_equal(
            audit,
            "manifest.model_snapshot.quantization",
            model_snapshot.get("quant_method"),
            "fp8",
        )

    runtime = manifest.get("runtime")
    if audit.check(
        "manifest.runtime.object",
        isinstance(runtime, dict),
        f"expected object, got {type(runtime).__name__}",
    ):
        deployment_budget = runtime.get("deployment_max_runtime_s")
        reserve = runtime.get("graceful_shutdown_reserve_s")
        soft_end_epoch = runtime.get("soft_end_epoch")
        notebook_start_epoch = manifest.get("notebook_start_epoch")
        audit.check(
            "manifest.runtime.deployment_budget",
            _same_number(deployment_budget, EXPECTED_DEPLOYMENT_MAX_RUNTIME_S),
            f"expected {EXPECTED_DEPLOYMENT_MAX_RUNTIME_S}, got {deployment_budget!r}",
        )
        audit.check(
            "manifest.runtime.shutdown_reserve",
            _same_number(reserve, EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S),
            f"expected {EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S}, got {reserve!r}",
        )
        expected_soft_offset = (
            EXPECTED_DEPLOYMENT_MAX_RUNTIME_S - EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S
        )
        audit.check(
            "manifest.runtime.soft_deadline",
            _is_finite_number(soft_end_epoch)
            and _is_finite_number(notebook_start_epoch)
            and math.isclose(
                float(soft_end_epoch) - float(notebook_start_epoch),
                expected_soft_offset,
                rel_tol=0.0,
                abs_tol=0.001,
            ),
            f"expected soft deadline offset {expected_soft_offset}, got "
            f"start={notebook_start_epoch!r}, end={soft_end_epoch!r}",
        )

    selected_count = manifest.get("selected_game_count")
    audit.check(
        "manifest.selected_game_count",
        _is_int(selected_count) and 0 < selected_count <= MAX_SELECTED_GAME_COUNT,
        f"expected integer in 1..{MAX_SELECTED_GAME_COUNT}, got {selected_count!r}",
    )
    if audit.mode == "preflight":
        audit.check(
            "manifest.preflight_game_count",
            _is_int(selected_count) and selected_count < EXPECTED_PUBLIC_GAME_COUNT,
            f"preflight must select 1..{EXPECTED_PUBLIC_GAME_COUNT - 1} games, got {selected_count!r}",
        )
    elif audit.mode == "full-offline":
        _check_equal(
            audit,
            "manifest.full_game_count",
            selected_count,
            EXPECTED_PUBLIC_GAME_COUNT,
        )

    offline_ids = manifest.get("offline_selected_game_ids")
    if expected_submission:
        audit.check(
            "manifest.offline_ids_absent",
            "offline_selected_game_ids" not in manifest,
            "submission manifest must not expose offline-selected ids",
        )
    else:
        valid_ids = (
            isinstance(offline_ids, list)
            and all(isinstance(game_id, str) and game_id for game_id in offline_ids)
            and len(set(offline_ids)) == len(offline_ids)
        )
        audit.check(
            "manifest.offline_ids",
            valid_ids and len(offline_ids) == selected_count,
            f"expected {selected_count!r} unique non-empty ids, got {offline_ids!r}",
        )
        if valid_ids:
            actual_ids = set(offline_ids)
            if audit.mode == "preflight":
                audit.check(
                    "manifest.preflight_public_ids",
                    actual_ids <= EXPECTED_PUBLIC_GAME_IDS,
                    f"unexpected preflight game ids: {sorted(actual_ids - EXPECTED_PUBLIC_GAME_IDS)}",
                )
            elif audit.mode == "full-offline":
                audit.check(
                    "manifest.full_public_ids",
                    actual_ids == EXPECTED_PUBLIC_GAME_IDS,
                    "full offline game ids do not match the pinned 25-game public set",
                )

    solver = manifest.get("solver")
    if audit.check(
        "manifest.solver.object",
        isinstance(solver, dict),
        f"expected object, got {type(solver).__name__}",
    ):
        mismatches = {
            key: {"expected": expected, "actual": solver.get(key)}
            for key, expected in EXPECTED_SOLVER_DEFAULTS.items()
            if not _same_number(solver.get(key), expected)
        }
        if audit.allow_solver_overrides:
            numeric = all(
                _is_finite_number(solver.get(key)) and float(solver[key]) > 0
                for key in EXPECTED_SOLVER_DEFAULTS
            )
            audit.check(
                "manifest.solver.positive_values",
                numeric,
                f"invalid solver values: {solver!r}",
            )
            if mismatches:
                env = manifest.get("env")
                recorded = isinstance(env, dict) and all(
                    _same_env_number(env.get(SOLVER_OVERRIDE_ENV[key]), solver.get(key))
                    for key in mismatches
                )
                audit.check(
                    "manifest.solver.overrides_recorded",
                    recorded,
                    "accepted solver mismatches must match their recorded TAAF_* environment overrides",
                )
                if recorded:
                    audit.warn(
                        f"solver overrides explicitly accepted: {json.dumps(mismatches, sort_keys=True)}"
                    )
        else:
            audit.check(
                "manifest.solver.defaults",
                not mismatches,
                f"solver mismatches: {mismatches}",
            )

    audit.summary["true_submission"] = true_submission
    audit.summary["selected_game_count"] = selected_count
    if isinstance(runtime, dict):
        audit.summary["runtime"] = {
            "deployment_max_runtime_s": runtime.get("deployment_max_runtime_s"),
            "graceful_shutdown_reserve_s": runtime.get("graceful_shutdown_reserve_s"),
        }
    return manifest


def _audit_benchmark(audit: Audit, manifest: dict[str, Any] | None) -> None:
    if audit.mode == "submission":
        path = audit.output_dir / "benchmark.json"
        audit.check(
            "benchmark.suppressed",
            not path.exists(),
            "submission mode must not expose hidden-run benchmark.json",
        )
        return
    benchmark = _read_json_object(audit, "benchmark.json")
    if benchmark is None:
        return

    _check_equal(
        audit, "benchmark.n_passes", benchmark.get("n_passes"), EXPECTED_N_PASSES
    )
    audit.check(
        "benchmark.end_time",
        isinstance(benchmark.get("end_time"), str),
        "expected terminal ISO timestamp string",
    )
    game_runs = benchmark.get("game_runs")
    if not audit.check(
        "benchmark.game_runs.array",
        isinstance(game_runs, list) and bool(game_runs),
        f"expected non-empty array, got {type(game_runs).__name__}",
    ):
        return
    if not audit.check(
        "benchmark.game_runs.bounded",
        len(game_runs) <= MAX_SELECTED_GAME_COUNT * EXPECTED_N_PASSES,
        f"expected at most {MAX_SELECTED_GAME_COUNT * EXPECTED_N_PASSES} runs, got {len(game_runs)}",
    ):
        return

    expected_count = (
        manifest.get("selected_game_count") if manifest is not None else None
    )
    target_count = expected_count
    audit.check(
        "benchmark.game_run_count",
        _is_int(target_count) and len(game_runs) == target_count * EXPECTED_N_PASSES,
        f"expected selected_count*n_passes={target_count!r}, got {len(game_runs)}",
    )

    run_ids: list[str] = []
    bad_run_indices: list[int] = []
    unfinished: list[dict[str, Any]] = []
    for index, run in enumerate(game_runs):
        if not isinstance(run, dict):
            bad_run_indices.append(index)
            continue
        game_id = run.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            bad_run_indices.append(index)
        else:
            run_ids.append(game_id)
        state = run.get("state")
        final_score = run.get("final_score")
        solver_note = run.get("solver_note")
        score_valid = (
            _is_finite_number(final_score) and 0.0 <= float(final_score) <= 100.0
        )
        has_error_note = isinstance(
            solver_note, str
        ) and solver_note.lower().startswith("error:")
        if state not in COMPLETED_GAME_STATES or not score_valid or has_error_note:
            unfinished.append(
                {
                    "index": index,
                    "game_id": game_id,
                    "state": state,
                    "final_score": final_score,
                }
            )
        actions_per_level = run.get("actions_per_level")
        history = run.get("history")
        number_of_levels = run.get("number_of_levels")
        if (
            not _is_int(number_of_levels)
            or number_of_levels <= 0
            or not isinstance(actions_per_level, list)
            or len(actions_per_level) != number_of_levels
            or not all(_is_int(value) and value >= 0 for value in actions_per_level)
            or not isinstance(history, list)
            or sum(actions_per_level) != len(history)
        ):
            bad_run_indices.append(index)

    audit.check(
        "benchmark.run_schema",
        not bad_run_indices,
        f"invalid run schema/action-history invariant at indices {sorted(set(bad_run_indices))[:20]}",
    )
    audit.check(
        "benchmark.terminal_runs",
        not unfinished,
        f"non-terminal or unscored runs: {unfinished[:20]}",
    )
    id_counts = Counter(run_ids)
    expected_repetitions = EXPECTED_N_PASSES
    audit.check(
        "benchmark.game_ids_unique_per_pass",
        bool(id_counts)
        and all(count == expected_repetitions for count in id_counts.values()),
        f"unexpected game-id repetitions: {dict(sorted(id_counts.items()))}",
    )

    if manifest is not None:
        offline_ids = manifest.get("offline_selected_game_ids")
        audit.check(
            "benchmark.game_ids_match_manifest",
            isinstance(offline_ids, list) and run_ids == offline_ids,
            f"benchmark ids {run_ids!r} do not match manifest ids {offline_ids!r}",
        )

    audit.summary["benchmark_game_runs"] = len(game_runs)
    audit.summary["benchmark_unique_games"] = len(id_counts)
    state_counts = Counter(
        str(run.get("state")) for run in game_runs if isinstance(run, dict)
    )
    known_states = ("won", "gave_up", "cancelled", "crashed", "playing", "not_started")
    audit.summary["game_states"] = {
        **{state: state_counts[state] for state in known_states if state_counts[state]},
        **(
            {
                "unknown": sum(state_counts.values())
                - sum(state_counts[state] for state in known_states)
            }
            if any(state not in known_states for state in state_counts)
            else {}
        ),
    }
    scores = [
        float(run["final_score"])
        for run in game_runs
        if isinstance(run, dict) and _is_finite_number(run.get("final_score"))
    ]
    if scores:
        audit.summary["mean_final_score"] = sum(scores) / len(scores)


def _audit_submission_health(audit: Audit, manifest: dict[str, Any] | None) -> None:
    path = audit.output_dir / "taaf_submission_health.json"
    if audit.mode != "submission":
        audit.check(
            "submission_health.absent",
            not path.exists(),
            "offline modes must not emit a competition submission health summary",
        )
        return
    health = _read_json_object(audit, "taaf_submission_health.json")
    if health is None:
        return
    allowed_keys = {
        "schema_version",
        "submission_variant",
        "n_passes",
        "selected_game_count",
        "game_run_count",
        "clean_game_run_count",
        "problem_game_run_count",
        "all_game_runs_completed_cleanly",
        "written_utc",
    }
    audit.check(
        "submission_health.aggregate_only",
        set(health) == allowed_keys,
        f"expected only aggregate health keys, got {sorted(health)}",
    )
    _check_equal(
        audit, "submission_health.schema_version", health.get("schema_version"), 1
    )
    _check_equal(
        audit,
        "submission_health.variant",
        health.get("submission_variant"),
        CANDIDATE_VARIANT,
    )
    _check_equal(
        audit, "submission_health.n_passes", health.get("n_passes"), EXPECTED_N_PASSES
    )
    selected_count = (
        manifest.get("selected_game_count") if manifest is not None else None
    )
    _check_equal(
        audit,
        "submission_health.selected_game_count",
        health.get("selected_game_count"),
        selected_count,
    )
    _check_equal(
        audit,
        "submission_health.game_run_count",
        health.get("game_run_count"),
        selected_count,
    )
    _check_equal(
        audit,
        "submission_health.clean_count",
        health.get("clean_game_run_count"),
        selected_count,
    )
    _check_equal(
        audit,
        "submission_health.problem_count",
        health.get("problem_game_run_count"),
        0,
    )
    _check_equal(
        audit,
        "submission_health.completed_cleanly",
        health.get("all_game_runs_completed_cleanly"),
        True,
    )
    audit.check(
        "submission_health.written_utc",
        isinstance(health.get("written_utc"), str),
        "expected UTC timestamp string",
    )
    audit.summary["submission_health"] = {
        "game_run_count": health.get("game_run_count"),
        "problem_game_run_count": health.get("problem_game_run_count"),
    }


def _audit_submission(audit: Audit) -> None:
    path = audit.output_dir / "submission.parquet"
    exists = audit.check(
        "submission.parquet",
        _is_regular_nonsymlink(path) and path.stat().st_size > 0,
        f"expected non-empty non-symlink regular file {path}",
    )
    if exists:
        try:
            with path.open("rb") as file:
                header = file.read(4)
                file.seek(-4, 2)
                trailer = file.read(4)
        except OSError as exc:
            audit.check(
                "submission.parquet.magic",
                False,
                f"could not inspect Parquet framing: {exc}",
            )
        else:
            audit.check(
                "submission.parquet.magic",
                header == b"PAR1" and trailer == b"PAR1",
                f"expected PAR1 header/trailer, got {header!r}/{trailer!r}",
            )
        audit.summary["submission_bytes"] = path.stat().st_size


def _audit_server_log(audit: Audit) -> None:
    path = audit.output_dir / "vllm-openai-server.log"
    if not audit.check(
        "vllm_log.exists",
        _is_regular_nonsymlink(path),
        f"expected non-symlink regular file {path}",
    ):
        return
    size = path.stat().st_size
    if not audit.check(
        "vllm_log.size",
        0 < size <= MAX_VLLM_LOG_BYTES,
        f"expected 1..{MAX_VLLM_LOG_BYTES} bytes, got {size}",
    ):
        return
    matches: dict[str, list[int]] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                for name, pattern in FATAL_LOG_PATTERNS:
                    if (
                        pattern.search(line)
                        and len(matches.setdefault(name, []))
                        < MAX_LOG_MATCHES_PER_PATTERN
                    ):
                        matches[name].append(line_number)
    except OSError as exc:
        audit.check("vllm_log.readable", False, f"could not read log: {exc}")
        return
    audit.check(
        "vllm_log.no_fatal_markers",
        not matches,
        f"fatal marker line numbers: {matches}",
    )
    audit.summary["vllm_log_bytes"] = size


def _audit_diagnostics(audit: Audit) -> None:
    path = audit.output_dir / "diagnostics.html"
    if audit.mode == "submission":
        audit.check(
            "diagnostics.suppressed",
            not path.exists(),
            "submission mode uses minimal diagnostics; diagnostics.html should be absent",
        )
    else:
        audit.check(
            "diagnostics.present",
            _is_regular_nonsymlink(path) and path.stat().st_size > 0,
            f"offline run must include non-empty non-symlink regular file {path}",
        )


def audit_output(
    output_dir: Path, *, mode: str, allow_solver_overrides: bool
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    audit = Audit(output_dir, mode=mode, allow_solver_overrides=allow_solver_overrides)
    audit.check(
        "output_dir",
        output_dir.is_dir() and not output_dir.is_symlink(),
        f"expected non-symlink directory {output_dir}",
    )
    if output_dir.is_dir() and not output_dir.is_symlink():
        manifest = _audit_manifest(audit)
        _audit_benchmark(audit, manifest)
        _audit_submission_health(audit, manifest)
        _audit_submission(audit)
        _audit_server_log(audit)
        _audit_diagnostics(audit)
    return audit.report()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=Path, help="Directory downloaded by `kaggle kernels output`."
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "full-offline", "submission"),
        required=True,
        help="Expected notebook execution mode and artifact contract.",
    )
    parser.add_argument(
        "--allow-solver-overrides",
        action="store_true",
        help="Accept positive non-default solver values and report them as warnings.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_output(
        args.output_dir,
        mode=args.mode,
        allow_solver_overrides=args.allow_solver_overrides,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
