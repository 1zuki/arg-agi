from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

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
AGENT_PATCH_SPEC = importlib.util.spec_from_file_location(
    "flash_agent_state_patch", ROOT / "scripts" / "flash_agent_state_patch.py"
)
AGENT_PATCH = importlib.util.module_from_spec(AGENT_PATCH_SPEC)
AGENT_PATCH_SPEC.loader.exec_module(AGENT_PATCH)


class FlashCoverageTests(unittest.TestCase):
    def test_pinned_agent_state_patch_is_digest_bound_and_preserves_markdown_reasoning(self):
        source = Path("/tmp/argagi-duck-src/full/tool_agent.py")
        if not source.is_file():
            self.skipTest("downloaded pinned tool-agent source is not present")
        patched = AGENT_PATCH.patch_tool_agent(source.read_bytes())
        tree = ast.parse(patched)
        needed = {"_normalize_summary_text", "_extract_labeled_blocks", "_extract_scientist_note"}
        parser_nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in needed
        ]
        namespace: dict = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(
                    body=[
                        ast.Import(names=[ast.alias(name="re")]),
                        ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
                        *parser_nodes,
                    ],
                    type_ignores=[],
                )),
                "patched-tool-agent-parser.py",
                "exec",
            ),
            namespace,
        )
        note = namespace["_extract_scientist_note"](
            "**World Model (v42):** newest scene\n**Plan:** take the verified route"
        )
        self.assertEqual(note["world_model"], "newest scene")
        self.assertEqual(note["current_plan"], "take the verified route")
        self.assertIn("self._update_summarized_knowledge_from_assistant(reasoning)", patched)
        with self.assertRaisesRegex(ValueError, "digest changed"):
            AGENT_PATCH.patch_tool_agent(source.read_bytes() + b"\n")

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

    def test_multi_batch_preflight_budget_scales_with_selected_games_and_concurrency(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(
            notebook, max_games=25, concurrency=12, runtime_seconds=1800,
            terminal_grace_seconds=120,
        )
        source = "\n".join(
            "".join(c.get("source", [])) for c in notebook["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn("offline_batch_count = max(1, (len(bm.games) + int(bm.solver.concurrency) - 1)", source)
        self.assertIn("FLASH_OFFLINE_MAX_RUNTIME_S * offline_batch_count + 600.0", source)
        self.assertIn("offline_batch_count={offline_batch_count}", source)
        assignments = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {"offline_batch_count", "budget"}
                    for target in node.targets)
        ]
        self.assertEqual(len(assignments), 2)
        code = compile(ast.Module(body=assignments, type_ignores=[]), "preflight-budget", "exec")
        for games, concurrency, runtime, batches, budget in (
            (25, 12, 1800, 3, 6000.0),
            (25, 28, 1800, 1, 2400.0),
            (12, 12, 1800, 1, 2400.0),
            (12, 28, 1800, 1, 2400.0),
            (24, 12, 1800, 2, 4200.0),
            (1, 8, 7920, 1, 8520.0),
            (25, 12, 0, 3, 32400.0),
            (25, 1, 1800, 25, 32400.0),
        ):
            with self.subTest(games=games, concurrency=concurrency, runtime=runtime):
                namespace = {
                    "bm": SimpleNamespace(games=list(range(games)),
                                          solver=SimpleNamespace(concurrency=concurrency)),
                    "target": SimpleNamespace(max_runtime_s=32400.0),
                    "FLASH_OFFLINE_MAX_RUNTIME_S": runtime,
                }
                exec(code, namespace)
                self.assertEqual(namespace["offline_batch_count"], batches)
                self.assertEqual(namespace["budget"], budget)

    def test_custom_preflight_settings_are_embedded_and_full_mode_stays_pinned(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(
            notebook,
            max_games=1,
            concurrency=8,
            game_id="tr87-cd924810",
            runtime_seconds=7920,
            include_agent_state_patch=True,
        )
        source = "\n".join(
            "".join(c.get("source", [])) for c in notebook["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn(
            'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "1"))',
            source,
        )
        self.assertIn(
            'FLASH_RUNTIME_CONCURRENCY = int(os.environ.get("FLASH_RUNTIME_CONCURRENCY", "8"))',
            source,
        )
        self.assertIn(
            'FLASH_OFFLINE_GAME_ID = os.environ.get("FLASH_OFFLINE_GAME_ID", "tr87-cd924810").strip()',
            source,
        )
        self.assertIn(
            'FLASH_OFFLINE_MAX_RUNTIME_S = 0.0 if TRUE_SUBMISSION else float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "7920"))',
            source,
        )
        self.assertIn(
            'PUBLIC25_DEADLINE origin=post_setup gameplay_budget_s={gameplay_budget_s}',
            source,
        )
        self.assertIn('terminal_grace_s={FLASH_TERMINAL_GRACE_S}', source)
        self.assertLess(
            source.index('vllm_watchdog.start_background'),
            source.index('PUBLIC25_DEADLINE origin=post_setup'),
        )
        self.assertIn(
            "bm.solver.concurrency = (FLASH_RUNTIME_CONCURRENCY if FLASH_RUNTIME_CONCURRENCY > 0",
            source,
        )
        self.assertIn(
            "not TRUE_SUBMISSION and (FLASH_OFFLINE_MAX_GAMES or FLASH_OFFLINE_GAME_ID)",
            source,
        )
        self.assertIn("FLASH_AGENT_STATE_PATCH", source)
        self.assertIn("FLASH_AGENT_OVERLAY_READY", source)
        self.assertIn("FLASH_AGENT_OVERLAY_REASSERTED", source)
        self.assertIn("FLASH_AGENT_OVERLAY_IMPORTED", source)
        self.assertLess(
            source.index("FLASH_AGENT_OVERLAY_READY"),
            source.index("taaf.kaggle: setup command:"),
        )
        self.assertLess(
            source.index("# Honour any PYTHONPATH a setup command exported."),
            source.index("FLASH_AGENT_OVERLAY_REASSERTED"),
        )
        self.assertLess(
            source.index("FLASH_AGENT_OVERLAY_REASSERTED"),
            source.index("FLASH_AGENT_OVERLAY_IMPORTED"),
        )

        full = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(full, max_games=None, concurrency=8)
        full_source = "\n".join(
            "".join(c.get("source", [])) for c in full["cells"]
            if c.get("cell_type") == "code"
        )
        self.assertIn(
            'FLASH_RUNTIME_CONCURRENCY = int(os.environ.get("FLASH_RUNTIME_CONCURRENCY", "8"))',
            full_source,
        )
        self.assertIn(
            "bm.solver.concurrency = (FLASH_RUNTIME_CONCURRENCY if FLASH_RUNTIME_CONCURRENCY > 0",
            full_source,
        )
        self.assertNotIn('PUBLIC25_DEADLINE origin=post_setup', full_source)

    def test_custom_analyzer_timeout_is_embedded_and_agent_overlay_can_be_disabled(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(
            notebook,
            max_games=12,
            analyzer_timeout=1200,
            include_agent_state_patch=False,
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn(
            'FLASH_ANALYZER_TIMEOUT = float(os.environ.get("FLASH_ANALYZER_TIMEOUT", "1200"))',
            source,
        )
        self.assertIn(
            "bm.solver.analyzer_timeout = (FLASH_ANALYZER_TIMEOUT",
            source,
        )
        self.assertNotIn("FLASH_AGENT_STATE_PATCH", source)
        self.assertNotIn("0.0 if TRUE_SUBMISSION else float(os.environ.get(\"FLASH_ANALYZER_TIMEOUT\"", source)

    def test_preflight_settings_reject_invalid_values(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        with self.assertRaisesRegex(ValueError, "max_games must be at most 25"):
            BUILDER._add_preflight_support(notebook, max_games=26, concurrency=8)
        with self.assertRaisesRegex(ValueError, "concurrency must be a positive integer"):
            BUILDER._add_preflight_support(notebook, max_games=8, concurrency=0)
        with self.assertRaisesRegex(ValueError, "pinned public game IDs"):
            BUILDER._add_preflight_support(notebook, max_games=1, game_id="not-a-game")
        with self.assertRaisesRegex(ValueError, "exactly one game"):
            BUILDER._add_preflight_support(notebook, max_games=2, game_id="tr87-cd924810")
        with self.assertRaisesRegex(ValueError, "terminal_grace_seconds"):
            BUILDER._add_preflight_support(notebook, max_games=1, terminal_grace_seconds=600)

    def test_terminal_grace_is_bounded_and_full_mode_rejects_it(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER._add_preflight_support(
            notebook,
            max_games=12,
            concurrency=12,
            runtime_seconds=1800,
            terminal_grace_seconds=120,
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn('FLASH_TERMINAL_GRACE_S = 0.0 if TRUE_SUBMISSION else float(os.environ.get("FLASH_TERMINAL_GRACE_S", "120"))', source)
        self.assertIn('not 0.0 <= FLASH_TERMINAL_GRACE_S < 600.0', source)

        args = type("Args", (), {
            "kernel_id": "izukia/flash-state-full-test",
            "mode": "full",
            "max_games": None,
            "concurrency": 12,
            "game_id": None,
            "runtime_seconds": None,
            "terminal_grace_seconds": 120,
            "title": None,
            "output_dir": ROOT / "build" / "test-unowned-output",
        })()
        with self.assertRaisesRegex(ValueError, "terminal-grace-seconds"):
            BUILDER.build(args)

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
