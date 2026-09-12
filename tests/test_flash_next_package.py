from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_flash_next_package", ROOT / "scripts/build_flash_next_package.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
PATCH_SPEC = importlib.util.spec_from_file_location(
    "flash_teardown_patch", ROOT / "scripts" / "flash_teardown_patch.py"
)
FLASH_PATCH = importlib.util.module_from_spec(PATCH_SPEC)
PATCH_SPEC.loader.exec_module(FLASH_PATCH)


class FlashCoverageTests(unittest.TestCase):
    def test_pinned_teardown_patch_is_digest_bound_and_compiles(self):
        source = ROOT / "build/flash-teardown-source/serving_teardown.py"
        if not source.is_file():
            self.skipTest("downloaded pinned teardown source is not present")
        patched = FLASH_PATCH.patch_flash_teardown(source.read_bytes())
        compile(patched, "patched-serving-teardown.py", "exec")
        self.assertIn("GPU_RELEASE_GRACE_SECONDS = 12.0", patched)
        with self.assertRaisesRegex(ValueError, "digest changed"):
            FLASH_PATCH.patch_flash_teardown(source.read_bytes() + b"\n")

    def test_coverage_uses_selected_ids_and_rejects_wrong_ids(self):
        for max_games, count in [(1, 1), (None, 25)]:
            with self.subTest(max_games=max_games):
                notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
                BUILDER._add_preflight_support(notebook, max_games=max_games)
                BUILDER._compile_notebook(notebook, "test")
                source = "\n".join(
                    "".join(c.get("source", []))
                    for c in notebook["cells"] if c.get("cell_type") == "code"
                )
                guards = [
                    node.test for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.If)
                    and "len(public_runs)" in ast.unparse(node.test)
                ]
                self.assertEqual(len(guards), 1)
                check = compile(ast.Expression(guards[0]), "coverage", "eval")
                ids = [str(i) for i in range(count)]
                env = dict(PUBLIC_GAME_IDS=ids, public_run_ids=ids, public_runs=ids)
                self.assertFalse(eval(check, env))
                self.assertTrue(eval(check, dict(env, public_runs=[])))
                self.assertTrue(eval(check, dict(env, public_run_ids=["wrong"])))

    def test_missing_coverage_anchor_fails_closed(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        for cell in notebook["cells"]:
            cell["source"] = "".join(cell.get("source", [])).replace(
                "len(public_runs) != 25", "len(public_runs) != 24"
            ).splitlines(keepends=True)
        with self.assertRaisesRegex(ValueError, "coverage=0"):
            BUILDER._add_preflight_support(notebook, max_games=1)

    def test_preflight_has_bounded_runtime_without_changing_full_mode(self):
        preflight = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(preflight, max_games=1)
        preflight_source = "\n".join(
            "".join(c.get("source", [])) for c in preflight["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("FLASH_OFFLINE_MAX_RUNTIME_S", preflight_source)
        self.assertIn("FLASH_OFFLINE_MAX_RUNTIME_S if FLASH_OFFLINE_MAX_RUNTIME_S > 0", preflight_source)

        full = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(full, max_games=None)
        full_source = "\n".join(
            "".join(c.get("source", [])) for c in full["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("bm.solver.max_runtime_s_per_game = 7920.0", full_source)

    def test_real_submission_leaves_deadline_to_kaggle_gateway(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(notebook, max_games=None)
        run_source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "soft_end =" in "".join(cell.get("source", []))
        )
        assignments = [
            node for node in ast.walk(ast.parse(run_source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "soft_end"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(value, ast.IfExp)
        self.assertEqual(ast.unparse(value.test), "TRUE_SUBMISSION")
        self.assertIsInstance(value.body, ast.Constant)
        self.assertIsNone(value.body.value)
        self.assertIn("seconds=budget - 600.0", ast.unparse(value.orelse))

    def test_teardown_uses_pinned_gpu_release_settle_patch(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(notebook, max_games=1)
        source = "\n".join(
            "".join(c.get("source", [])) for c in notebook["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("gpu-release-settle-v1", source)
        self.assertIn("FLASH_TEARDOWN_PATH", source)
        self.assertIn("GPU_RELEASE_GRACE_SECONDS", source)
        self.assertNotIn("for attempt in range(3)", source)
