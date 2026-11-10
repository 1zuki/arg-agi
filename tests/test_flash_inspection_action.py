from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/flash-baseline-source-20260920/tool_agent.py"
PATCH_PATH = ROOT / "scripts/flash_inspection_action_patch.py"
BUILDER_PATH = ROOT / "scripts/build_pulled_flash_inspection_action_candidate.py"
AUDIT_PATH = ROOT / "scripts/audit_flash_output.py"
sys.path.insert(0, str(ROOT / "scripts"))


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PATCH = load(PATCH_PATH, "flash_inspection_action_patch")
BUILDER = load(BUILDER_PATH, "pulled_inspection_builder")
AUDIT = load(AUDIT_PATH, "inspection_audit")


def _tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }


def _fake_imports() -> dict[str, types.ModuleType]:
    requests = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        pass

    requests.RequestException = RequestException
    requests.HTTPError = HTTPError
    requests.post = lambda *args, **kwargs: None

    inference = types.ModuleType("inference")
    inference.__path__ = []
    agent = types.ModuleType("inference.agent")
    agent.__path__ = []
    utils = types.ModuleType("inference.utils")
    utils.__path__ = []

    action_names = types.ModuleType("inference.agent.action_names")
    action_names.to_engine_action = lambda value: value
    action_names.to_model_action = lambda value: value

    prompts = types.ModuleType("inference.agent.prompts")
    for name in (
        "COMPACT_TOOL_SESSION_ADDENDUM",
        "GAME_OVERVIEW_ADDENDUM",
        "PYTHON_ADDENDUM",
        "STRUCTURED_RUNTIME_STATE_ADDENDUM",
        "MULTIMODAL_CONTEXT_ADDENDUM",
        "TOOL_CALL_FORMAT_GUIDANCE",
        "VISUAL_GAME_ADDENDUM",
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


def load_patched_agent():
    source = PATCH.patch_tool_agent(SOURCE.read_bytes())
    module = types.ModuleType("inspection_tool_agent")
    module.__file__ = str(SOURCE)
    with patch.dict(sys.modules, _fake_imports()):
        sys.modules[module.__name__] = module
        exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


class InspectionActionPatchTests(unittest.TestCase):
    def test_patch_is_digest_bound_and_compiles(self):
        source = SOURCE.read_bytes()
        patched = PATCH.patch_tool_agent(source)
        self.assertEqual(hashlib.sha256(source).hexdigest(), PATCH.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(patched.encode()).hexdigest(), AUDIT.INSPECTION_ACTION_PATCH_HASH)
        self.assertEqual(PATCH.PATCH_NAME, AUDIT.INSPECTION_ACTION_PATCH_NAME)
        self.assertIn('yielded_control_reason == "turn_time_budget"', patched)
        self.assertIn('append_inspection_action_followup()', patched)
        self.assertEqual(patched.count("inspection_action_followup_forced"), 1)
        self.assertEqual(patched.count("forced_inspection_followup_used = True"), 1)
        compile(patched, "patched-inspection-action-tool-agent.py", "exec")
        with self.assertRaisesRegex(ValueError, "digest changed"):
            PATCH.patch_tool_agent(source + b"\n")

    def _run_analyze(self, responses: list, dispatch_steps: list[bool]):
        module = load_patched_agent()
        agent = object.__new__(module.ToolAgent)
        agent._model = module.AnalyzerModelConfig(
            provider="test", base_url="http://127.0.0.1", model_id="test-model"
        )
        agent._timeout = None
        agent._api_key = ""
        agent._tool_steps = 5
        agent._python_timeout = 1
        agent._yield_seconds = 1.0
        agent._max_output_tokens = 128
        agent._reply_reserve_tokens = 128
        agent._tool_output_tokens = 64
        agent._tool_output_chars = 256
        agent._save_request_logs = False
        agent._system_prompt = "system"
        agent._request_safety_margin_tokens = 1
        agent._context_budget_tokens = 4096
        agent._history_messages = []
        agent._session_runtime_dir = None
        agent._session_total_tokens = 0
        agent._session_generated_tokens = 0
        agent._step_env_callback = None
        agent._current_valid_actions = []
        agent._last_step_summary = None
        agent._last_action_result = None
        agent._summarized_knowledge = module._empty_world_model()
        agent._ensure_session = MethodType(lambda self, path: None, agent)
        agent._tools = MethodType(lambda self, path: [], agent)
        agent._trim_messages_for_context = MethodType(
            lambda self, messages, **kwargs: messages, agent
        )
        agent._build_user_prompt = MethodType(
            lambda self, action_num, **kwargs: "initial prompt", agent
        )
        agent._build_user_message = MethodType(
            lambda self, prompt, frame: {"role": "user", "content": prompt}, agent
        )
        agent._persistent_history_messages = MethodType(
            lambda self, messages, **kwargs: messages, agent
        )

        response_index = 0
        dispatch_index = 0
        clock = SimpleNamespace(value=0.0)

        def chat_completion(self, messages, **kwargs):
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return module._ChatCompletionResult(message=response, finish_reason="tool_calls")

        def dispatch_tool(self, state_path, name, arguments):
            nonlocal dispatch_index
            step = dispatch_steps[dispatch_index]
            dispatch_index += 1
            if not step:
                clock.value += 2.0
            return module._ToolDispatchResult("tool result", step_executed=step)

        agent._chat_completion = MethodType(chat_completion, agent)
        agent._dispatch_tool = MethodType(dispatch_tool, agent)

        with tempfile.TemporaryDirectory() as temp, patch.object(
            module, "_write_prompt_log_snapshot", lambda *args, **kwargs: None
        ), patch.object(module.time, "monotonic", lambda: clock.value):
            state_path = Path(temp) / "state.json"
            state_path.write_text("{}")
            transcript_path = Path(temp) / "transcript.txt"
            result = agent.analyze(
                state_path,
                0,
                valid_actions=["UP"],
                transcript_path=transcript_path,
                analysis_step=1,
                should_stop=lambda: False,
            )
            transcript = transcript_path.read_text()
        return result, response_index, transcript

    def test_timeout_after_inspection_gets_one_bounded_action_followup(self):
        result, responses, transcript = self._run_analyze(
            [
                {"tool_calls": [_tool_call("inspect", "print(current_frame)")]},
                {"tool_calls": [_tool_call("act", "action(['UP'])")]},
            ],
            [False, True],
        )
        self.assertTrue(result.step_executed)
        self.assertFalse(result.yielded_control)
        self.assertEqual(responses, 2)
        self.assertEqual(transcript.count("inspection_action_followup_forced"), 1)

    def test_followup_cannot_chain_when_it_still_inspects(self):
        result, responses, transcript = self._run_analyze(
            [
                {"tool_calls": [_tool_call("inspect-1", "print(current_frame)")]},
                {"tool_calls": [_tool_call("inspect-2", "print(history)")]},
            ],
            [False, False],
        )
        self.assertFalse(result.step_executed)
        self.assertTrue(result.yielded_control)
        self.assertEqual(responses, 2)
        self.assertEqual(transcript.count("inspection_action_followup_forced"), 1)

    def test_ordinary_inspection_without_timeout_is_not_forced(self):
        module = load_patched_agent()
        agent = object.__new__(module.ToolAgent)
        # The patch's control-flow contract is verified structurally here; the
        # behavioral timeout path above covers the timer-reset execution.
        source = PATCH.patch_tool_agent(SOURCE.read_bytes())
        self.assertNotIn(
            'if inspection_only_python_call:\n'
            '                    forced_inspection_followup_used = True',
            source,
        )
        self.assertIn(
            'yielded_control_reason == "turn_time_budget"\n'
            '                        and inspection_only_python_call',
            source,
        )

    def test_candidate_builder_isolated_from_prior_experiments(self):
        notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
        BUILDER.patch_notebook(
            notebook,
            mode="preflight",
            max_games=1,
            runtime_seconds=1800,
            terminal_grace_seconds=120,
        )
        BUILDER._compile_notebook(notebook, "inspection-action-test")
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("FLASH_INSPECTION_ACTION_PATCH", source)
        self.assertIn("FLASH_INSPECTION_ACTION_OVERLAY_IMPORTED", source)
        self.assertNotIn("FLASH_ACTION7_PATCH", source)
        self.assertNotIn("FLASH_DIRECT_RETRY_PATCH", source)
        self.assertNotIn("FLASH_AGENT_STATE_PATCH", source)
        self.assertIn("not a verified score improvement", "".join(notebook["cells"][0]["source"]).lower())


if __name__ == "__main__":
    unittest.main()
