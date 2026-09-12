from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "arc-agi.ipynb"


def _load_patch_function():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def _apply_candidate_patch" in "".join(cell.get("source", []))
    )
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_replace_exact", "_apply_candidate_patch"}
    ]
    namespace = {"Path": Path}
    exec(  # noqa: S102 - execute the notebook's extracted patch functions in a test namespace.
        compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK_PATH), "exec"),
        namespace,
    )
    return namespace["_apply_candidate_patch"]


def _write_patch_fixture(root: Path) -> None:
    agent_dir = root / "src/ARC3-Inference/inference/agent"
    framework_dir = root / "src/ARC3-Inference/inference/framework"
    agent_dir.mkdir(parents=True)
    framework_dir.mkdir(parents=True)

    (framework_dir / "solver.py").write_text(
        """class Session:
    def play(self):
        try:
            while True:
                if True:
                    self._execute_auto_reset()
                    continue
        finally:
            pass

    def step_env(self, requested_actions):
        executed_payloads = []
        requested_displays = []
        stop_reason = None
        for batch_index, action in enumerate(requested_actions, start=1):
            if self.should_stop():
                break
        final_payload = {}
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        self.write_viewer_payload()

    def _execute_auto_reset(self) -> None:
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
""",
        encoding="utf-8",
    )
    (agent_dir / "tool_agent.py").write_text(
        """from typing import Any

class ToolAgent:
    def render(self, payload):
        compact = {}
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            compact[timing_key] = payload.get(timing_key)
        return compact

    @property
    def total_tokens(self) -> int:
        return 0
""",
        encoding="utf-8",
    )
    (agent_dir / "action_names.py").write_text(
        """ENGINE_TO_MODEL_ACTION = {
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}

def to_engine_action(name):
    raw = str(name or "").strip().upper()
    if raw in ENGINE_TO_MODEL_ACTION:
        return raw
    return MODEL_TO_ENGINE_ACTION.get(raw)

def to_model_action(name):
    raw = str(name or "").strip().upper()
    return ENGINE_TO_MODEL_ACTION.get(raw, raw)
""",
        encoding="utf-8",
    )


class CandidatePatchTests(unittest.TestCase):
    def test_patch_maps_undo_and_synchronizes_auto_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_patch_fixture(root)

            _load_patch_function()(root)

            solver = (
                root / "src/ARC3-Inference/inference/framework/solver.py"
            ).read_text()
            tool = (
                root / "src/ARC3-Inference/inference/agent/tool_agent.py"
            ).read_text()
            action_names = (
                root / "src/ARC3-Inference/inference/agent/action_names.py"
            ).read_text()
            self.assertIn('"ACTION7": "UNDO"', action_names)
            self.assertIn(
                'record_auto_reset = getattr(self.analyzer, "record_auto_reset", None)',
                solver,
            )
            self.assertIn(
                "if _is_engine_game_over(self.game):\n                        break",
                solver,
            )
            self.assertIn("def record_auto_reset(self, payload: dict[str, Any])", tool)
            self.assertIn("self._history_messages = []", tool)

            compile(solver, "solver.py", "exec")
            compile(tool, "tool_agent.py", "exec")
            compile(action_names, "action_names.py", "exec")
            action_namespace: dict[str, object] = {}
            exec(  # noqa: S102 - execute the isolated mapping fixture in a test namespace.
                compile(action_names, "action_names.py", "exec"),
                action_namespace,
            )
            self.assertEqual(
                action_namespace["to_model_action"]("ACTION7"),
                "UNDO",
            )
            self.assertEqual(
                action_namespace["to_engine_action"]("UNDO"),
                "ACTION7",
            )
            self.assertEqual(
                action_namespace["to_engine_action"]("ACTION7"),
                "ACTION7",
            )

    def test_patch_fails_closed_after_first_application(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_patch_fixture(root)
            apply_patch = _load_patch_function()
            apply_patch(root)

            with self.assertRaisesRegex(RuntimeError, "matched 0 times"):
                apply_patch(root)


if __name__ == "__main__":
    unittest.main()
