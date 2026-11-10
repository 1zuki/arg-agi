from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTROLLER = load_script("arc_ensemble_controller")
PATCH = load_script("arc_ensemble_patch")
BUILDER = load_script("build_arc_ensemble_package")


def action_result(
    action: str = "UP",
    *,
    board_changed: bool = True,
    valid_actions: list[str] | None = None,
    executed: bool = True,
    reward: float = 0.0,
    done: bool = False,
    level_completed: bool = False,
    game_over: bool = False,
    run_complete: bool = False,
    requested_count: int = 1,
    executed_count: int = 1,
    stop_reason: str | None = None,
    error: str | None = None,
) -> dict:
    result = {
        "executed": executed,
        "action_display": action,
        "board_changed": board_changed,
        "valid_actions": ["UP", "DOWN"] if valid_actions is None else valid_actions,
        "reward": reward,
        "done": done,
        "level_completed": level_completed,
        "game_over": game_over,
        "run_complete": run_complete,
        "requested_count": requested_count,
        "executed_count": executed_count,
    }
    if stop_reason is not None:
        result["stop_reason"] = stop_reason
    if error is not None:
        result["error"] = error
    return result


def signal(result: dict | None, *, frame: str = "frame-1", level: int = 1, **kwargs):
    return CONTROLLER.TurnSignal(
        action_result=result,
        current_frame={"ascii": frame, "level": level},
        valid_actions=(result or {}).get("valid_actions", ["UP", "DOWN"]),
        level=level,
        score=0,
        state="PLAYING",
        **kwargs,
    )


class FakeEnvironment:
    def __init__(self) -> None:
        self.controller = CONTROLLER.BoundedRecoveryController(
            CONTROLLER.ControllerConfig(
                no_progress_window=3,
                invalid_action_window=2,
                stale_terminal_window=2,
                tool_exhaustion_window=2,
                max_recovery_actions=2,
                max_recovery_turns=3,
            )
        )
        self.controller.reset_game("game-a")

    def observe(self, result: dict | None, **kwargs):
        return self.controller.record_turn(signal(result, **kwargs))


class EnsembleControllerTests(unittest.TestCase):
    def test_repeated_valid_action_that_changes_board_stays_flash(self):
        env = FakeEnvironment()
        for _ in range(5):
            reason = env.observe(action_result(board_changed=True))
        self.assertIsNone(reason)
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)
        self.assertEqual(env.controller.handoff_count, 0)

    def test_confirmed_repeated_no_progress_triggers_one_handoff(self):
        env = FakeEnvironment()
        reason = None
        for _ in range(3):
            reason = env.observe(action_result(board_changed=False))
        self.assertEqual(reason, CONTROLLER.HandoffReason.REPEAT_NO_CHANGE_GUARD)
        snapshot = env.controller.handoff(
            reason,
            game_id="game-a",
            current_frame={"ascii": "fresh"},
            valid_actions=["UP", "DOWN"],
            level=1,
            levels_completed=0,
            score=0,
            state="PLAYING",
        )
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.HYBRID)
        self.assertEqual(snapshot.handoff_reason, "repeat_no_change_guard")
        self.assertEqual(env.controller.handoff_count, 1)

    def test_changed_valid_action_sets_block_repeat_guard(self):
        env = FakeEnvironment()
        for actions in (["UP", "DOWN"], ["UP"], ["UP", "DOWN"]):
            reason = env.observe(action_result(board_changed=False, valid_actions=list(actions)))
        self.assertIsNone(reason)
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)

    def test_invalid_actions_require_repeated_rejection(self):
        env = FakeEnvironment()
        rejected = lambda: action_result(
            executed=False,
            board_changed=False,
            executed_count=0,
            stop_reason="invalid_action",
        )
        self.assertIsNone(env.observe(rejected()))
        self.assertEqual(
            env.observe(rejected()), CONTROLLER.HandoffReason.INVALID_ACTION_FAILURE
        )

    def test_terminal_transition_and_level_completion_do_not_trigger(self):
        env = FakeEnvironment()
        terminal = action_result(
            executed=False,
            board_changed=False,
            executed_count=0,
            done=True,
            run_complete=True,
            valid_actions=[],
        )
        self.assertIsNone(env.observe(terminal))
        # Repeated terminal-incompatible actions are an explicit stale-state signal.
        self.assertEqual(
            env.observe(terminal), CONTROLLER.HandoffReason.STALE_TERMINAL_STATE
        )
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)

        env = FakeEnvironment()
        completed = action_result(level_completed=True, reward=1.0)
        self.assertIsNone(env.observe(completed))
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)

    def test_missing_progress_metadata_is_not_no_progress(self):
        env = FakeEnvironment()
        incomplete = action_result(board_changed=False)
        incomplete.pop("board_changed")
        incomplete.pop("reward")
        for _ in range(5):
            self.assertIsNone(env.observe(incomplete))
        self.assertEqual(env.controller.no_progress_streak, 0)
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)

    def test_tool_loop_exhaustion_is_bounded(self):
        env = FakeEnvironment()
        self.assertIsNone(env.observe(None, retryable_failure=True))
        self.assertEqual(
            env.observe(None, retryable_failure=True),
            CONTROLLER.HandoffReason.TOOL_LOOP_EXHAUSTION,
        )

    def test_timeout_pressure_requires_confirmed_no_progress(self):
        env = FakeEnvironment()
        reason = None
        for _ in range(3):
            reason = env.observe(action_result(board_changed=False), remaining_seconds=60.0)
        self.assertEqual(reason, CONTROLLER.HandoffReason.REPEAT_NO_CHANGE_GUARD)

    def test_no_return_and_recovery_exhaustion(self):
        env = FakeEnvironment()
        for _ in range(3):
            reason = env.observe(action_result(board_changed=False))
        env.controller.handoff(
            reason,
            game_id="game-a",
            current_frame={"ascii": "fresh"},
            valid_actions=["UP"],
            level=1,
            levels_completed=0,
            score=0,
            state="PLAYING",
        )
        env.observe(action_result(action="DOWN"), frame="recovery")
        env.controller.note_recovery_action()
        env.observe(action_result(action="UP"), frame="recovery")
        env.controller.note_recovery_action()
        self.assertTrue(env.controller.recovery_exhausted)
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.HYBRID)
        self.assertEqual(env.controller.handoff_count, 1)
        self.assertIsNone(env.controller.record_turn(signal(None, frame="recovery")))

    def test_reset_and_level_transition_require_fresh_observation(self):
        env = FakeEnvironment()
        env.controller.record_reset_transition(level=2)
        self.assertTrue(env.controller.fresh_observation_required)
        with self.assertRaisesRegex(RuntimeError, "fresh post-transition"):
            env.controller.handoff(
                CONTROLLER.HandoffReason.TOOL_LOOP_EXHAUSTION,
                game_id="game-a",
                current_frame={"ascii": "stale"},
                valid_actions=["UP"],
                level=2,
                levels_completed=1,
                score=1,
                state="PLAYING",
            )
        env.observe(None, frame="new-level", level=2)
        self.assertFalse(env.controller.fresh_observation_required)

    def test_one_way_handoff_and_game_reset_clear_memory(self):
        env = FakeEnvironment()
        for _ in range(3):
            reason = env.observe(action_result(board_changed=False))
        env.controller.handoff(
            reason,
            game_id="game-a",
            current_frame={"ascii": "fresh"},
            valid_actions=["UP"],
            level=1,
            levels_completed=0,
            score=0,
            state="PLAYING",
        )
        with self.assertRaises(RuntimeError):
            env.controller.handoff(
                reason,
                game_id="game-a",
                current_frame={"ascii": "fresh-2"},
                valid_actions=["UP"],
                level=1,
                levels_completed=0,
                score=0,
                state="PLAYING",
            )
        env.controller.record_mechanic_fact("private game fact")
        env.controller.reset_game("game-b")
        self.assertEqual(env.controller.owner, CONTROLLER.AgentOwner.FLASH)
        self.assertEqual(env.controller.handoff_count, 0)
        self.assertEqual(env.controller.no_progress_streak, 0)
        self.assertEqual(list(env.controller.memory.mechanic_facts), [])
        self.assertEqual(env.controller.memory.game_id, "game-b")

    def test_snapshot_is_bounded_and_privacy_safe(self):
        env = FakeEnvironment()
        for _ in range(3):
            reason = env.observe(action_result(board_changed=False))
        snapshot = env.controller.handoff(
            reason,
            game_id="opaque-id",
            current_frame={
                "ascii": "x" * 20_000,
                "transcript": "must not cross boundary",
                "nested": {"reasoning": "must not cross boundary", "ok": True},
            },
            valid_actions=["UP"],
            level=1,
            levels_completed=0,
            score=0,
            state="PLAYING",
        )
        encoded = json.dumps(snapshot.as_dict())
        self.assertLess(len(encoded), 17_000)
        self.assertNotIn("transcript", encoded)
        self.assertNotIn("reasoning", encoded)

    def test_health_summary_contains_aggregate_fields_only(self):
        env = FakeEnvironment()
        summary = env.controller.health_summary()
        self.assertEqual(summary["agent_owner"], "flash")
        self.assertIn("hybrid_handoffs", summary)
        self.assertNotIn("game_id", summary)
        self.assertNotIn("current_frame", summary)


class EnsembleSourceIsolationTests(unittest.TestCase):
    def test_active_solver_digest_and_overlay_compile(self):
        source = (ROOT / "build/flash-baseline-source-20260920/solver.py").read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), PATCH.SOLVER_SOURCE_SHA256)
        patched = PATCH.patch_solver(source)
        compile(patched, "arc_ensemble_solver.py", "exec")
        with self.assertRaisesRegex(ValueError, "digest changed"):
            PATCH.patch_solver(source + b"\n")

    def test_public_hybrid_reference_is_not_mislabeled(self):
        source = Path("/tmp/hybrid-pull-codex/arc-agi-3-hybrid-repl-agent.py")
        if not source.is_file():
            self.skipTest("read-only Hybrid pull is not present")
        report = PATCH.inspect_hybrid_reference(source.read_bytes())
        self.assertFalse(report["adapter_available"])
        self.assertEqual(report["role"], "launcher_only")
        self.assertEqual(report["source"], "pulled_public")

    def test_existing_experiments_are_not_imported_by_overlay(self):
        patched = PATCH.patch_solver(
            (ROOT / "build/flash-baseline-source-20260920/solver.py").read_bytes()
        )
        for forbidden in (
            "ACTION7",
            "state_overlay",
            "direct_retry",
            "inspection_action",
            "zero_change_repeat_guard",
        ):
            self.assertNotIn(forbidden, patched)

    def test_generated_package_isolated_and_metadata_is_fail_closed(self):
        package = ROOT / "build/arc-ensemble-controller-preflight"
        notebook = package / "arc-agi3-flash-next-mtp-ensemble-controller.ipynb"
        metadata = json.loads((package / "kernel-metadata.json").read_text())
        receipt = json.loads((package / "hybrid-reference-receipt.json").read_text())
        source_receipt = json.loads((package / "ensemble-source-receipt.json").read_text())
        self.assertTrue(notebook.is_file())
        self.assertEqual(receipt["adapter_available"], False)
        self.assertEqual(source_receipt["active_solver_sha256"], PATCH.SOLVER_SOURCE_SHA256)
        self.assertEqual(source_receipt["hybrid_adapter_available"], False)
        self.assertEqual(metadata["adapter_available"], False)
        self.assertEqual(metadata["protected_incumbent"]["submission"], "56170646")
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in json.loads(notebook.read_text())["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("FLASH_ENSEMBLE_OVERLAY_READY", source)
        self.assertIn("no genuine Hybrid adapter is installed", source)
        for forbidden in (
            "FLASH_ACTION7",
            "FLASH_AGENT_STATE_PATCH",
            "FLASH_DIRECT_RETRY",
            "FLASH_INSPECTION_ACTION",
            "FLASH_ZERO_CHANGE",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"', source)
        self.assertIn('"TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192"', source)
        BUILDER._compile_notebook(json.loads(notebook.read_text()), str(notebook))


if __name__ == "__main__":
    unittest.main()
