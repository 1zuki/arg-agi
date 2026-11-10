from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_combined_harness_patch",
    ROOT / "scripts/check_combined_harness_patch.py",
)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


class CombinedHarnessPatchTests(unittest.TestCase):
    def test_extract_patch_matches_embedded_payload(self) -> None:
        patch = CHECKER.extract_patch(
            ROOT / "arc-agi-3-milestone-2-solution-combined.ipynb"
        )
        self.assertTrue(patch.startswith(b"diff --git "))
        self.assertEqual(
            hashlib.sha256(patch).hexdigest(),
            "04bf5743d10064bbb9d97e4e63632e6e54287beea3a50841f0a67c207e77c81e",
        )

    def test_extract_patch_rejects_wrong_magic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = Path(temp_dir) / "bad.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {"cell_type": "code", "source": ["%not-the-patch\n"]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must begin"):
                CHECKER.extract_patch(notebook, cell_index=0)

    def test_normalize_legacy_seed_removes_exactly_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool_agent = (
                Path(temp_dir)
                / "ARC3-Inference"
                / "inference"
                / "agent"
                / "tool_agent.py"
            )
            tool_agent.parent.mkdir(parents=True)
            tool_agent.write_text(
                "before\n"
                + CHECKER.LEGACY_SEED_LINE
                + "after\n",
                encoding="utf-8",
            )
            self.assertEqual(CHECKER.normalize_legacy_seed(Path(temp_dir)), 1)
            self.assertEqual(tool_agent.read_text(encoding="utf-8"), "before\nafter\n")

    def test_normalize_legacy_seed_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tool_agent = (
                Path(temp_dir)
                / "ARC3-Inference"
                / "inference"
                / "agent"
                / "tool_agent.py"
            )
            tool_agent.parent.mkdir(parents=True)
            tool_agent.write_text(
                CHECKER.LEGACY_SEED_LINE * 2,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                CHECKER.normalize_legacy_seed(Path(temp_dir))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
