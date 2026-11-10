"""Contract tests for the optional Skyfall shelf smoke."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "arc3_skyfall_shelf_smoke.py"


class SkyfallShelfSmokeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        spec = importlib.util.spec_from_file_location("arc3_skyfall_shelf_smoke", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_script_compiles_without_arcengine_import_at_module_load(self) -> None:
        compile(self.source, str(SCRIPT), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_empty_shelf_fails_loudly(self) -> None:
        with self.assertRaises(self.module.ShelfSmokeError):
            self.module.audit_shelf(Path("/definitely/missing/skyfall-shelf"))

    def test_action_contract_keeps_action7_and_click_payload(self) -> None:
        self.assertEqual(self.module._parse_action('["ACTION7", {}]', label="x"), (7, {}))
        self.assertEqual(
            self.module._parse_action('["CLICK", {"x": 2, "y": 3}]', label="x"),
            (6, {"x": 2, "y": 3}),
        )
        with self.assertRaises(self.module.ShelfSmokeError):
            self.module._parse_action('["CLICK", {}]', label="x")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
