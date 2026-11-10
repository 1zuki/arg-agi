"""CPU-only regression checks for the active ARC-AGI-3 combined notebook.

These checks cover the notebook artifact and its runnable Python syntax. They
do not load the model, start SGLang, play games, or claim a competition score.
"""

from __future__ import annotations

import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "arc-agi-3-milestone-2-solution-combined.ipynb"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


class CombinedNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = [_source(cell) for cell in cls.notebook["cells"]]
        cls.code = [
            source
            for cell, source in zip(cls.notebook["cells"], cls.cells)
            if cell.get("cell_type") == "code"
        ]
        cls.active_text = "\n".join(cls.cells)

    def test_notebook_shape_and_json(self) -> None:
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertIsInstance(self.notebook.get("cells"), list)
        self.assertGreater(len(self.notebook["cells"]), 0)

    def test_runnable_python_cells_compile_on_cpu(self) -> None:
        for index, source in enumerate(self.code):
            # The patch cell is an IPython writefile cell rather than Python.
            if source.lstrip().startswith("%%"):
                continue
            try:
                compile(
                    source,
                    f"{NOTEBOOK.name}:cell-{index}",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
            except SyntaxError as exc:  # pragma: no cover - failure detail
                self.fail(f"cell {index} does not compile: {exc}")

    def test_legacy_v6_patch_markers_are_absent(self) -> None:
        for marker in (
            "LOCAL_ANALYZER_SEED",
            "Patched combined winner profile v6",
        ):
            self.assertNotIn(marker, self.active_text)

    def test_harness_patch_normalizes_legacy_seed_before_apply(self) -> None:
        setup = self.cells[4]
        self.assertIn("_legacy_seed_line = \"            seed=\" + _legacy_seed_name", setup)
        self.assertIn("_legacy_seed_count = _tool_agent_source.count(_legacy_seed_line)", setup)
        self.assertIn("_tool_agent_source.replace(_legacy_seed_line, \"\", 1)", setup)
        self.assertIn('subprocess.run([\"git\", \"apply\", \"--verbose\", \"/kaggle/harness-changes.patch\"]', setup)
        self.assertLess(
            setup.index("_tool_agent_source.replace(_legacy_seed_line"),
            setup.index('subprocess.run([\"git\", \"apply\"'),
        )

    def test_offline_path_emits_only_an_empty_submission_scaffold(self) -> None:
        offline_cell = next(
            source for source in self.code
            if "Offline validation is diagnostic only" in source
        )
        self.assertIn("if not TRUE_SUBMISSION:", offline_cell)
        self.assertIn('"row_id": pd.Series(dtype="string")', offline_cell)
        self.assertIn('"game_id": pd.Series(dtype="string")', offline_cell)
        self.assertIn('"end_of_game": pd.Series(dtype="bool")', offline_cell)
        self.assertIn('"score": pd.Series(dtype="int64")', offline_cell)
        self.assertIn('"submission.parquet"', offline_cell)
        self.assertIn("no competition predictions were produced", offline_cell)

    def test_post_run_contract_validates_benchmark_and_parquet(self) -> None:
        benchmark_cell = next(
            source for source in self.code if "_validate_benchmark_output" in source
        )
        for marker in (
            "benchmark.json must contain a non-empty game_runs list",
            "state not in {\"won\", \"gave_up\"}",
            "_validate_parquet_output",
            "expected_columns = [\"row_id\", \"game_id\", \"end_of_game\", \"score\"]",
            "parquet_file.read(columns=expected_columns).to_pylist()",
            "duplicate row_id values",
            "require_rows=TRUE_SUBMISSION",
            "combined_run_contract.json",
            "Offline run completed without benchmark.json",
        ):
            self.assertIn(marker, benchmark_cell)

    def _contract_helpers(self):
        source = next(
            source for source in self.code if "_validate_benchmark_output" in source
        )
        tree = ast.parse(source)
        names = {"_read_contract_json", "_validate_benchmark_output", "_validate_parquet_output"}
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        namespace = {"Path": Path, "json": json, "math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "contract-helpers", "exec"), namespace)
        return namespace

    def test_post_run_contract_accepts_valid_benchmark_and_parquet(self) -> None:
        helpers = self._contract_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark.json").write_text(
                json.dumps(
                    {
                        "n_passes": 1,
                        "game_runs": [
                            {
                                "game_id": "game-1",
                                "state": "won",
                                "final_score": 4.5,
                                "number_of_levels": 1,
                                "actions_per_level": [2],
                                "history": [{}, {}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = helpers["_validate_benchmark_output"](root / "benchmark.json")
            self.assertEqual(summary["game_runs"], 1)
            self.assertEqual(summary["unique_game_ids"], 1)
            (root / "submission.parquet").write_bytes(b"PAR1dataPAR1")

            class _Schema:
                names = ["row_id", "game_id", "end_of_game", "score"]

            class _Metadata:
                num_rows = 1

            class _Table:
                def to_pylist(self):
                    return [{
                        "row_id": "game-1_0",
                        "game_id": "game-1",
                        "end_of_game": True,
                        "score": 1,
                    }]

            class _ParquetFile:
                schema_arrow = _Schema()
                metadata = _Metadata()

                def read(self, *, columns):
                    self.columns = columns
                    return _Table()

            self.assertEqual(
                helpers["_validate_parquet_output"](
                    root / "submission.parquet",
                    require_rows=True,
                    parquet_file_factory=lambda _: _ParquetFile(),
                )["framed"],
                True,
            )

    def test_post_run_contract_rejects_wrong_parquet_schema(self) -> None:
        helpers = self._contract_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.parquet"
            path.write_bytes(b"PAR1dataPAR1")

            class _Schema:
                names = ["game_id", "score"]

            class _Metadata:
                num_rows = 1

            class _ParquetFile:
                schema_arrow = _Schema()
                metadata = _Metadata()

            with self.assertRaisesRegex(RuntimeError, "columns must be"):
                helpers["_validate_parquet_output"](
                    path,
                    require_rows=True,
                    parquet_file_factory=lambda _: _ParquetFile(),
                )

    def test_post_run_contract_rejects_invalid_parquet_values(self) -> None:
        helpers = self._contract_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.parquet"
            path.write_bytes(b"PAR1dataPAR1")

            class _Schema:
                names = ["row_id", "game_id", "end_of_game", "score"]

            class _Metadata:
                num_rows = 2

            class _Table:
                def to_pylist(self):
                    return [
                        {"row_id": "duplicate", "game_id": "game-1", "end_of_game": True, "score": 1.0},
                        {"row_id": "duplicate", "game_id": "game-2", "end_of_game": False, "score": float("nan")},
                    ]

            class _ParquetFile:
                schema_arrow = _Schema()
                metadata = _Metadata()

                def read(self, *, columns):
                    return _Table()

            with self.assertRaisesRegex(RuntimeError, "invalid score"):
                helpers["_validate_parquet_output"](
                    path,
                    require_rows=True,
                    parquet_file_factory=lambda _: _ParquetFile(),
                )

    def test_post_run_contract_rejects_nonterminal_benchmark(self) -> None:
        helpers = self._contract_helpers()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "n_passes": 1,
                        "game_runs": [
                            {
                                "game_id": "game-1",
                                "state": "cancelled",
                                "final_score": 0.0,
                                "number_of_levels": 1,
                                "actions_per_level": [0],
                                "history": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                helpers["_validate_benchmark_output"](path)

    def test_rolling_summary_profile_is_opt_in(self) -> None:
        setup = self.cells[4]
        self.assertIn("'ARC3_REASONING_HISTORY_KEY': 'reasoning_content'", setup)
        for setting in (
            "'ARC3_SUMMARY_INTERVAL_TOKENS':",
            "'ARC3_SUMMARY_REPLACES_HISTORY':",
            "'ARC3_SUMMARY_TURN_CONTEXT':",
            "'ARC3_SUMMARY_ENABLE_THINKING':",
            "'ARC3_SUMMARY_MAX_GEN':",
        ):
            self.assertNotIn(setting, setup)
        self.assertIn("_summary_replaces_history", self.cells[2])

    def test_summary_replacement_is_gated_on_usable_text(self) -> None:
        patch = self.cells[2]
        self.assertIn("_SUMMARY_MIN_USABLE_CHARS = 500", patch)
        self.assertIn("if len(text) < _SUMMARY_MIN_USABLE_CHARS:", patch)
        self.assertIn("if _summary_replaces_history():", patch)
        self.assertIn("self._history_messages = [request, reply]", patch)

    def test_runtime_caps_are_explicit_and_comment_is_not_stale(self) -> None:
        self.assertNotIn("48*60 #7920.0", self.active_text)
        self.assertIn("bm.solver.max_runtime_s_per_game = 532*60  # competition branch", self.active_text)
        self.assertIn("bm.solver.max_runtime_s_per_game = 25*60  # offline branch", self.active_text)

    def test_startup_timeout_matches_protected_profile(self) -> None:
        self.assertIn("SERVER_STARTUP_TIMEOUT = 12 * 60", self.cells[4])

    def test_offline_selection_and_solver_overrides_are_documented(self) -> None:
        for marker in (
            "TAAF_OFFLINE_GAMES",
            "TAAF_OFFLINE_MAX_GAMES",
            "TAAF_BENCHMARK_CONCURRENCY",
            "TAAF_ANALYZER_TIMEOUT",
            "TAAF_MAX_ACTIONS_PER_GAME",
            "TAAF_MAX_RUNTIME_S_PER_GAME",
            "cannot exceed the notebook budget",
        ):
            self.assertIn(marker, self.active_text)

    def test_offline_selection_is_exact_bounded_and_fail_closed(self) -> None:
        source = next(source for source in self.code if "def _select_offline_game_ids" in source)
        tree = ast.parse(source)
        node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_select_offline_game_ids"
        )
        namespace = {"os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "offline-selection", "exec"), namespace)
        select = namespace["_select_offline_game_ids"]
        with patch.dict(os.environ, {"TAAF_OFFLINE_GAMES": "game-3, game-1", "TAAF_OFFLINE_MAX_GAMES": "1"}, clear=True):
            self.assertEqual(select(["game-1", "game-2", "game-3"]), ["game-1"])
        with patch.dict(os.environ, {"TAAF_OFFLINE_GAMES": "missing"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Unknown TAAF_OFFLINE_GAMES"):
                select(["game-1"])
        with patch.dict(os.environ, {"TAAF_OFFLINE_MAX_GAMES": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                select(["game-1"])

    def test_solver_override_parser_rejects_invalid_values(self) -> None:
        source = next(source for source in self.code if "def _positive_env_override" in source)
        tree = ast.parse(source)
        node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_positive_env_override"
        )
        namespace = {"os": os, "math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "solver-overrides", "exec"), namespace)
        parse = namespace["_positive_env_override"]
        with patch.dict(os.environ, {"TEST_OVERRIDE": "12"}, clear=True):
            self.assertEqual(parse("TEST_OVERRIDE", integer=True), 12)
        with patch.dict(os.environ, {"TEST_OVERRIDE": "nan"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive finite number"):
                parse("TEST_OVERRIDE")
        with patch.dict(os.environ, {"TEST_OVERRIDE": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                parse("TEST_OVERRIDE", integer=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
