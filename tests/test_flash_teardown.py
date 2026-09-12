"""Behavior tests against the downloaded, hash-verified public teardown.

Source fixture (read-only; no GPU required):
uvx --from kaggle kaggle datasets download \
  -d keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 \
  -f serving_teardown.py -p build/flash-teardown-source
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "flash_teardown_patch", ROOT / "scripts/flash_teardown_patch.py"
)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)
SOURCE = ROOT / "build/flash-teardown-source/serving_teardown.py"


@unittest.skipUnless(SOURCE.is_file(), "download the pinned teardown fixture first")
class FlashTeardownTests(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("teardown_under_test")
        exec(compile(PATCHER.patch_flash_teardown(SOURCE.read_bytes()),
                     "patched_teardown.py", "exec"), self.module.__dict__)

    def simulate(self, *, never_release=False, query_error=False,
                 capture_fails=False, corrupt_artifact=False,
                 port_reopens=False, conflict=False, cpu_survivor=False):
        m = self.module
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        def gpu_rows():
            clock[0] += 0.05
            if query_error:
                return [{"query_error": "simulated nvidia-smi failure"}]
            if never_release or clock[0] < 5.0:
                return [{"pid": 238, "start_ticks": 1,
                         "process_name": "VLLM::Worker"}]
            return []

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "vllm-server-identity.json").write_text("{}")
            (root / "submission.parquet").write_bytes(b"PAR1fixturePAR1")
            (root / "score.json").write_text('{"score": 0}')
            scan = {
                "authorized": {}, "authorized_records": [], "suspect_records": [],
                "root_conflict": {"pid": 238} if conflict else None,
                "saved_conflicts": [],
            }

            def capture(metrics, models, result):
                if capture_fails:
                    result["final_metrics_error"] = "simulated capture failure"
                    return
                metrics.write_bytes(b"# mock metrics\nrequests_total 1\n")
                models.write_text('{"data": []}')
                result["final_metrics_sha256"] = m.sha256_file(metrics)

            def capture_log(*args):
                if corrupt_artifact:
                    (root / "submission.parquet").write_bytes(b"changed")

            capture_mock = Mock(side_effect=capture)
            stop_mock = Mock(return_value=(scan, {(238, 1)}))
            with patch.multiple(
                m,
                validate_working_dir=lambda: root,
                current_boot_id=lambda: "test-boot",
                validate_identity_document=lambda *args: ({"workers": []}, []),
                capture_endpoints=capture_mock,
                capture_log=capture_log,
                stop_owned_processes=stop_mock,
                owned_process_table=lambda identity: {},
                scan_ownership=lambda *args: scan,
                global_marker_records=lambda: [{"pid": 999}] if cpu_survivor else [],
                port_open=lambda: port_reopens and clock[0] > 4,
                gpu_rows=gpu_rows,
                time=types.SimpleNamespace(
                    monotonic=lambda: clock[0], sleep=sleep, time=time.time,
                ),
            ), contextlib.redirect_stdout(io.StringIO()):
                if any((never_release, query_error, capture_fails, corrupt_artifact,
                        port_reopens, conflict, cpu_survivor)):
                    with self.assertRaisesRegex(RuntimeError, "bounded terminal gate"):
                        m.main()
                else:
                    m.main()
            result = json.loads((root / "vllm-server-teardown.json").read_text())
            capture_mock.assert_called_once()
            stop_mock.assert_called_once()
            self.assertLessEqual(clock[0], 12.0)
            return result

    def test_delayed_gpu_release_passes_without_recapturing_metrics(self):
        result = self.simulate()
        self.assertTrue(result["shutdown_ok"])
        self.assertTrue(result["final_metrics_preserved"])
        self.assertTrue(result["required_artifacts_preserved"])
        self.assertGreater(result["gpu_release_poll_count"], 1)
        self.assertGreaterEqual(result["gpu_release_wait_seconds"], 5.0)

    def test_survivor_times_out_instead_of_relaxing_gate(self):
        result = self.simulate(never_release=True)
        self.assertFalse(result["shutdown_ok"])
        self.assertTrue(result["vllm_gpu_rows_after"])

    def test_query_failure_is_not_an_empty_gpu(self):
        result = self.simulate(query_error=True)
        self.assertTrue(result["gpu_query_error_after"])
        self.assertEqual(result["gpu_release_poll_count"], 1)

    def test_missing_metrics_still_fails(self):
        self.assertFalse(self.simulate(capture_fails=True)["final_metrics_preserved"])

    def test_modified_submission_still_fails(self):
        self.assertFalse(self.simulate(corrupt_artifact=True)["required_artifacts_preserved"])

    def test_reopened_port_still_fails(self):
        self.assertFalse(self.simulate(port_reopens=True)["port_closed"])

    def test_ownership_conflict_still_fails(self):
        self.assertFalse(self.simulate(conflict=True)["shutdown_ok"])

    def test_cpu_marker_survivor_still_fails(self):
        self.assertFalse(self.simulate(cpu_survivor=True)["shutdown_ok"])

    def test_upstream_process_identity_self_tests_still_pass(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.module.run_self_tests()
