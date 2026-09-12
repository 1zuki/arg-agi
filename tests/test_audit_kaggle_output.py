from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_kaggle_output", REPO_ROOT / "scripts" / "audit_kaggle_output.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _run(game_id: str, *, state: str = "gave_up") -> dict:
    return {
        "game_id": game_id,
        "number_of_levels": 1,
        "base_actions_per_level": [10],
        "hint": None,
        "state": state,
        "history": [],
        "record_intermediate_states": True,
        "actions_per_level": [0],
        "levels_completed": 0,
        "final_score": 0.0,
        "solver_note": None,
        "solver_analysis_html": None,
        "final_generated_tokens": 1,
        "final_uncached_input_tokens": 0,
        "final_wallclock_seconds": 1.0,
        "started_at": "2026-08-18T00:00:00",
    }


def _manifest(game_ids: list[str], *, submission: bool = False) -> dict:
    notebook_start_epoch = 1_787_011_200.1234567
    soft_end_epoch = round(
        notebook_start_epoch
        + AUDITOR.EXPECTED_DEPLOYMENT_MAX_RUNTIME_S
        - AUDITOR.EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S,
        6,
    )
    return {
        "milestone": "benchmark_completed",
        "outcome": "completed",
        "submission_variant": AUDITOR.CANDIDATE_VARIANT,
        "model": {
            "kaggle_dataset_ref": AUDITOR.MODEL_DATASET_REF,
            "huggingface_ref": AUDITOR.MODEL_HF_REF,
            "huggingface_revision": AUDITOR.MODEL_HF_REVISION,
            "served_name": AUDITOR.MODEL_HF_REF,
            "reasoning_effort": AUDITOR.MODEL_REASONING_EFFORT,
        },
        "batch_checkpoint_limit": 8,
        "adapter_fixes": AUDITOR.EXPECTED_ADAPTER_FIXES,
        "expected_dataset_versions": AUDITOR.EXPECTED_DATASET_VERSIONS,
        "TRUE_SUBMISSION": submission,
        "notebook_start_epoch": notebook_start_epoch,
        "runtime": {
            "deployment_max_runtime_s": AUDITOR.EXPECTED_DEPLOYMENT_MAX_RUNTIME_S,
            "graceful_shutdown_reserve_s": AUDITOR.EXPECTED_GRACEFUL_SHUTDOWN_RESERVE_S,
            "soft_end_epoch": soft_end_epoch,
        },
        "env": {},
        "bundle_patch": {
            "status": "verified_and_applied",
            "source_tree": AUDITOR.EXPECTED_SOURCE_TREE,
            "patched_tree": AUDITOR.EXPECTED_PATCHED_TREE,
        },
        "wheelhouse": {
            "status": "inventory_and_sha256_verified",
            "file_count": 180,
            "payload_sha256_count": 178,
            "core_versions": {
                "vllm": "0.19.0",
                "torch": "2.10.0",
                "transformers": "4.57.6",
            },
        },
        "model_snapshot": {
            "status": "inventory_and_crc32_verified",
            "file_count": 80,
            "crc32_count": 77,
            "crc32_correction_count": AUDITOR.EXPECTED_MODEL_CRC32_CORRECTION_COUNT,
            "weight_shard_count": 66,
            "quant_method": "fp8",
        },
        "solver": dict(AUDITOR.EXPECTED_SOLVER_DEFAULTS),
        "selected_game_count": len(game_ids),
        **({} if submission else {"offline_selected_game_ids": game_ids}),
    }


def _write_fixture(
    root: Path, game_ids: list[str], *, submission: bool = False
) -> None:
    (root / "taaf_run_manifest.json").write_text(
        json.dumps(_manifest(game_ids, submission=submission)), encoding="utf-8"
    )
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "label": "duck-harness",
                "n_passes": 1,
                "solver_label": "duck-harness",
                "start_time": "2026-08-18T00:00:00",
                "end_time": "2026-08-18T01:00:00",
                "game_weights": None,
                "game_runs": [_run(game_id) for game_id in game_ids],
            }
        ),
        encoding="utf-8",
    )
    (root / "submission.parquet").write_bytes(b"PAR1synthetic-testPAR1")
    (root / "vllm-openai-server.log").write_text(
        "INFO Application startup complete.\nINFO graceful server shutdown.\n",
        encoding="utf-8",
    )
    if submission:
        (root / "benchmark.json").unlink()
        (root / "taaf_submission_health.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "submission_variant": AUDITOR.CANDIDATE_VARIANT,
                    "n_passes": 1,
                    "selected_game_count": len(game_ids),
                    "game_run_count": len(game_ids),
                    "clean_game_run_count": len(game_ids),
                    "problem_game_run_count": 0,
                    "all_game_runs_completed_cleanly": True,
                    "written_utc": "2026-08-18T01:00:00Z",
                }
            ),
            encoding="utf-8",
        )
    else:
        (root / "diagnostics.html").write_text("<html>ok</html>\n", encoding="utf-8")


class AuditKaggleOutputTests(unittest.TestCase):
    def test_preflight_passes_without_mutating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            before = {path.name: path.read_bytes() for path in root.iterdir()}

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            after = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(before, after)

    def test_submission_accepts_gateway_selected_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_fixture(root, ["hidden-a", "hidden-b"], submission=True)

            report = AUDITOR.audit_output(
                root, mode="submission", allow_solver_overrides=False
            )

            self.assertTrue(report["passed"], report["errors"])

    def test_runtime_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            manifest_path = root / "taaf_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["soft_end_epoch"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any(
                    "manifest.runtime.soft_deadline" in error
                    for error in report["errors"]
                )
            )

    def test_model_crc_correction_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            manifest_path = root / "taaf_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_snapshot"].pop("crc32_correction_count")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any(
                    "manifest.model_snapshot.crc_correction_count" in error
                    for error in report["errors"]
                )
            )

    def test_adapter_fix_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            manifest_path = root / "taaf_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["adapter_fixes"]["action7_model_label"] = "ACTION7"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("manifest.adapter_fixes" in error for error in report["errors"])
            )

    def test_crashed_game_fails_even_with_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            benchmark_path = root / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            benchmark["game_runs"][0] = _run(game_ids[0], state="crashed")
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("benchmark.terminal_runs" in error for error in report["errors"])
            )

    def test_solver_override_requires_flag_and_recorded_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            manifest_path = root / "taaf_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["solver"]["concurrency"] = 14
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            strict = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )
            unrecorded = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=True
            )
            manifest["env"]["TAAF_BENCHMARK_CONCURRENCY"] = "14"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            accepted = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=True
            )

            self.assertFalse(strict["passed"])
            self.assertFalse(unrecorded["passed"])
            self.assertTrue(accepted["passed"], accepted["errors"])
            self.assertTrue(accepted["warnings"])

    def test_required_artifact_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            real_log = root / "real-vllm.log"
            (root / "vllm-openai-server.log").replace(real_log)
            (root / "vllm-openai-server.log").symlink_to(real_log)

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("vllm_log.exists" in error for error in report["errors"])
            )

    def test_malformed_solver_override_fails_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            manifest_path = root / "taaf_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["solver"].pop("analyzer_timeout")
            manifest["env"]["TAAF_ANALYZER_TIMEOUT"] = "900"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=True
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("manifest.solver" in error for error in report["errors"])
            )

    def test_fatal_server_log_marker_fails_without_echoing_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = [min(AUDITOR.EXPECTED_PUBLIC_GAME_IDS)]
            _write_fixture(root, game_ids)
            secret_marker = "do-not-copy-this-private-log-content"
            (root / "vllm-openai-server.log").write_text(
                f"Traceback (most recent call last): {secret_marker}\n",
                encoding="utf-8",
            )

            report = AUDITOR.audit_output(
                root, mode="preflight", allow_solver_overrides=False
            )
            rendered = json.dumps(report)

            self.assertFalse(report["passed"])
            self.assertIn("python_traceback", rendered)
            self.assertNotIn(secret_marker, rendered)

    def test_submission_health_rejects_hidden_detail_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_ids = ["hidden-a", "hidden-b"]
            _write_fixture(root, game_ids, submission=True)
            health_path = root / "taaf_submission_health.json"
            health = json.loads(health_path.read_text(encoding="utf-8"))
            health["game_ids"] = game_ids
            health_path.write_text(json.dumps(health), encoding="utf-8")

            report = AUDITOR.audit_output(
                root, mode="submission", allow_solver_overrides=False
            )

            self.assertFalse(report["passed"])
            self.assertTrue(
                any(
                    "submission_health.aggregate_only" in error
                    for error in report["errors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
