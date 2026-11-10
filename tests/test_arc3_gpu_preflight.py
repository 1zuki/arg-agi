from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "arc3_gpu_preflight.py"
SPEC = importlib.util.spec_from_file_location("arc3_gpu_preflight", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Arc3GpuPreflightTests(unittest.TestCase):
    def test_no_gpu_is_not_ready(self) -> None:
        with patch.object(MODULE, "_nvidia_smi", return_value=([], "no GPU")):
            report = MODULE.run_preflight()
        self.assertFalse(report["ready"])
        self.assertEqual(report["probe_error"], "no GPU")
        self.assertEqual(report["side_effects"], [])

    def test_small_gpu_is_rejected(self) -> None:
        gpu = MODULE.GpuRecord("test", 6141, 5000)
        with patch.object(MODULE, "_nvidia_smi", return_value=([gpu], None)):
            report = MODULE.run_preflight()
        self.assertFalse(report["ready"])
        self.assertEqual(report["eligible_gpus"], [])

    def test_eligible_gpu_passes_threshold_gate(self) -> None:
        gpu = MODULE.GpuRecord("test", 49152, 30000)
        with patch.object(MODULE, "_nvidia_smi", return_value=([gpu], None)):
            report = MODULE.run_preflight()
        self.assertTrue(report["ready"])
        self.assertEqual(report["eligible_gpus"][0]["name"], "test")

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.run_preflight(minimum_total_mib=0)


if __name__ == "__main__":
    unittest.main()
