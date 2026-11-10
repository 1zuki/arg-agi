"""Contract tests for the optional ARC-AGI-3 CPU environment smoke."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "arc3_cpu_environment_smoke.py"


class Arc3CpuSmokeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        spec = importlib.util.spec_from_file_location("arc3_cpu_environment_smoke", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_script_compiles_without_runtime_dependency(self) -> None:
        compile(self.source, str(SCRIPT), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_complex_action_uses_coordinates(self) -> None:
        self.assertIn("GameAction.ACTION6", self.source)
        self.assertIn('"x": int(width // 2)', self.source)
        self.assertIn('"y": int(height // 2)', self.source)

    def test_invalid_action_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.module.run_smoke(Path("environment_files"), actions_per_game=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
