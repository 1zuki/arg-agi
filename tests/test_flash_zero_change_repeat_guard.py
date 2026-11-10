from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/flash-baseline-source-20260920/tool_agent.py"
PATCH_PATH = ROOT / "scripts/flash_zero_change_repeat_guard_patch.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_imports() -> dict[str, types.ModuleType]:
    requests = types.ModuleType("requests")
    requests.RequestException = type("RequestException", (Exception,), {})
    requests.HTTPError = type("HTTPError", (requests.RequestException,), {})
    requests.post = lambda *args, **kwargs: None

    inference = types.ModuleType("inference")
    inference.__path__ = []
    agent = types.ModuleType("inference.agent")
    agent.__path__ = []
    utils = types.ModuleType("inference.utils")
    utils.__path__ = []
    action_names = types.ModuleType("inference.agent.action_names")
    action_names.to_engine_action = lambda value: str(value).strip().upper()
    action_names.to_model_action = lambda value: str(value).strip().upper()
    prompts = types.ModuleType("inference.agent.prompts")
    for name in (
        "COMPACT_TOOL_SESSION_ADDENDUM", "GAME_OVERVIEW_ADDENDUM", "PYTHON_ADDENDUM",
        "STRUCTURED_RUNTIME_STATE_ADDENDUM", "MULTIMODAL_CONTEXT_ADDENDUM",
        "TOOL_CALL_FORMAT_GUIDANCE", "VISUAL_GAME_ADDENDUM",
    ):
        setattr(prompts, name, "")
    vision = types.ModuleType("inference.agent.vision_context")
    vision.current_grid_image_enabled = lambda: False
    vision.current_grid_image_part = lambda frame: None
    sandbox = types.ModuleType("inference.agent.python_tool_sandbox")
    sandbox.run_sandboxed_python = lambda **kwargs: {"stdout": "", "action_results": []}
    runtime = types.ModuleType("inference.agent.runtime_state")
    runtime.Frame = object
    runtime.HistoryEntry = object
    runtime.RUNTIME_STATE_FILENAME = "runtime_state.json"
    runtime.load_runtime_state = lambda path: (None, [])
    openai_compat = types.ModuleType("inference.utils.openai_compat")
    openai_compat.build_chat_payload = lambda **kwargs: kwargs
    openai_compat.build_headers = lambda **kwargs: {}
    return {
        "requests": requests,
        "inference": inference,
        "inference.agent": agent,
        "inference.utils": utils,
        "inference.agent.action_names": action_names,
        "inference.agent.prompts": prompts,
        "inference.agent.vision_context": vision,
        "inference.agent.python_tool_sandbox": sandbox,
        "inference.agent.runtime_state": runtime,
        "inference.utils.openai_compat": openai_compat,
    }


class ZeroChangeRepeatGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file():
            raise unittest.SkipTest("pinned source fixture not downloaded")
        cls.patch = load(PATCH_PATH, "zero_change_repeat_guard_patch")
        cls.source = SOURCE.read_bytes()
        cls.patched = cls.patch.patch_tool_agent(cls.source)

    def test_patch_is_digest_bound_and_stable(self):
        self.assertEqual(hashlib.sha256(self.source).hexdigest(), self.patch.SOURCE_SHA256)
        self.assertEqual(
            hashlib.sha256(self.patched.encode()).hexdigest(),
            "ec358aef45e3d22ee07112a946fcd6f48b7022dde1c01840d966b2087262168c",
        )
        with self.assertRaisesRegex(ValueError, "digest changed"):
            self.patch.patch_tool_agent(self.source + b"\n")

    def _agent(self, raw_payload):
        module = types.ModuleType("zero_change_guard_agent")
        module.__file__ = str(SOURCE)
        imports = fake_imports()
        with patch.dict(sys.modules, imports):
            sys.modules[module.__name__] = module
            exec(compile(self.patched, str(SOURCE), "exec"), module.__dict__)

        agent = object.__new__(module.ToolAgent)
        agent._current_valid_actions = ["RIGHT"]
        agent._last_action_result = {
            "executed": True,
            "action_num": 4,
            "level": 1,
            "score": 0,
            "reward": 0.0,
            "state": "NOT_FINISHED",
            "valid_actions": ["RIGHT"],
            "board_changed": False,
            "done": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "executed_actions": ["RIGHT"],
        }
        agent._step_env_callback = lambda arguments: raw_payload
        agent._last_step_summary = None
        agent._session_runtime_dir = None
        agent._python_timeout = 1
        agent._tool_output_tokens = 64
        agent._tool_output_chars = 256
        agent._ensure_session = MethodType(lambda self, path: None, agent)
        return module, agent

    def _run(self, action_code, raw_payload, *, current_valid_actions=None, previous_updates=None):
        module, agent = self._agent(raw_payload)
        if current_valid_actions is not None:
            agent._current_valid_actions = list(current_valid_actions)
        if previous_updates:
            agent._last_action_result.update(previous_updates)

        def fake_sandbox(**kwargs):
            if action_code == "repeat":
                actions = [{"action": "RIGHT"}]
            elif action_code == "multi":
                actions = [{"action": "RIGHT"}, {"action": "RIGHT"}]
            else:
                actions = [{"action": "LEFT"}]
            result = kwargs["action_handler"](actions)
            return {"stdout": "", "action_results": [result["action_result"]]}

        module.run_sandboxed_python = fake_sandbox
        module._ascii_frame_view_payload = lambda frame: {}
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state.json"
            state.write_text("{}")
            result = agent._run_python_tool(state, {"code": "action(['RIGHT'])"})
        return json.loads(result.content)

    def test_identical_zero_change_singleton_is_blocked_once(self):
        payload = self._run(
            "repeat",
            {
                "executed": True,
                "action_num": 5,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["RIGHT"],
                "board_changed": True,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "RIGHT",
            },
        )
        self.assertFalse(payload["result"]["executed"])
        self.assertEqual(payload["result"]["stop_reason"], "repeat_no_change_guard")

    def test_different_action_is_not_blocked(self):
        payload = self._run(
            "different",
            {
                "executed": True,
                "action_num": 5,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["LEFT"],
                "board_changed": True,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "LEFT",
            },
        )
        self.assertTrue(payload["result"]["executed"])

    def test_changed_valid_actions_are_not_blocked(self):
        payload = self._run(
            "repeat",
            {
                "executed": True,
                "action_num": 5,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["LEFT"],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "RIGHT",
            },
            current_valid_actions=["LEFT"],
        )
        self.assertTrue(payload["result"]["executed"])

    def test_terminal_or_rewarded_previous_result_is_not_blocked(self):
        for field, value in (("done", True), ("level_completed", True), ("game_over", True), ("run_complete", True), ("reward", 1.0)):
            with self.subTest(field=field):
                payload = self._run(
                    "repeat",
                    {
                        "executed": True,
                        "action_num": 5,
                        "level": 1,
                        "score": 0,
                        "reward": 0.0,
                        "state": "NOT_FINISHED",
                        "valid_actions": ["RIGHT"],
                        "board_changed": False,
                        "done": False,
                        "level_completed": False,
                        "game_over": False,
                        "run_complete": False,
                        "action_display": "RIGHT",
                    },
                    previous_updates={field: value},
                )
                self.assertTrue(payload["result"]["executed"])

    def test_multi_action_request_is_never_blocked(self):
        payload = self._run(
            "multi",
            {
                "executed": True,
                "action_num": 5,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["RIGHT"],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "RIGHT",
            },
        )
        self.assertTrue(payload["result"]["executed"])

    def test_unknown_progress_metadata_is_not_treated_as_no_change(self):
        payload = self._run(
            "repeat",
            {
                "executed": True,
                "action_num": 5,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["RIGHT"],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "RIGHT",
            },
            previous_updates={"board_changed": None},
        )
        self.assertTrue(payload["result"]["executed"])
