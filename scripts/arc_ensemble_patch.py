"""Digest-bound solver overlay for the conservative Flash ensemble."""

from __future__ import annotations

import hashlib


PATCH_NAME = "flash-primary-hybrid-recovery-controller-v1"
SOLVER_SOURCE_SHA256 = (
    "65ddfcdbe8c24fdc31246ff018b51469827b21c38995243a36a818adbed33370"
)
HYBRID_REFERENCE_SHA256 = (
    "7da69071ea8d21bcf7d7b8b3179cc847220d07601c706be1c8b0b53dcca29d73"
)
PULLED_HYBRID_REFERENCE_SHA256 = (
    "4aab5ab4b1eea5e53bb54f76b33298017331a3e3735fdd8e2d6fec8e62fc409b"
)


def inspect_hybrid_reference(source: bytes) -> dict[str, object]:
    """Prove the pulled public reference is launcher-only, not an adapter."""
    digest = hashlib.sha256(source).hexdigest()
    if digest not in {HYBRID_REFERENCE_SHA256, PULLED_HYBRID_REFERENCE_SHA256}:
        raise ValueError("Hybrid reference digest changed; refusing to classify it.")
    text = source.decode("utf-8")
    if "bm.run(" not in text or "class Hybrid" in text:
        raise ValueError("Hybrid reference shape changed; manual adapter review required.")
    return {
        "sha256": digest,
        "role": "launcher_only",
        "adapter_available": False,
        "source": "downloads" if digest == HYBRID_REFERENCE_SHA256 else "pulled_public",
    }


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"ensemble solver anchor {label!r} matched {count} times")
    return text.replace(old, new, 1)


def patch_solver(source: bytes) -> str:
    """Add the controller seam to the exact active solver source."""
    if hashlib.sha256(source).hexdigest() != SOLVER_SOURCE_SHA256:
        raise ValueError("Flash solver source digest changed; refusing to patch.")
    text = source.decode("utf-8")

    text = _replace_once(
        text,
        "from typing import Any, Callable\n",
        """from typing import Any, Callable

from arc_ensemble_controller import (
    BoundedRecoveryController,
    TurnSignal,
)
""",
        "controller import",
    )
    text = _replace_once(
        text,
        "    _viewer_events_flushed: int = field(default=0, init=False, repr=False)\n",
        """    _viewer_events_flushed: int = field(default=0, init=False, repr=False)
    ensemble_controller: BoundedRecoveryController = field(
        default_factory=BoundedRecoveryController, init=False, repr=False
    )
    ensemble_last_action_result: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
""",
        "session fields",
    )
    text = _replace_once(
        text,
        "        self.token_baseline = _analyzer_reported_tokens(self.analyzer)\n",
        "        self.token_baseline = _analyzer_reported_tokens(self.analyzer)\n"
        "        self.ensemble_controller.reset_game(run.game_id)\n",
        "session reset",
    )
    text = _replace_once(
        text,
        "                self.write_runtime_state()\n                transcript_before = self._read_transcript_bytes()\n",
        "                self.write_runtime_state()\n"
        "                self.ensemble_last_action_result = None\n"
        "                transcript_before = self._read_transcript_bytes()\n",
        "turn result reset",
    )
    text = _replace_once(
        text,
        "                if result is None:\n                    raise RuntimeError(\"Analyzer did not return a result.\")\n",
        """                if result is None:
                    self._record_ensemble_turn(
                        step_executed=False,
                        retryable_failure=True,
                        yielded_control=False,
                        tool_turns_exhausted=False,
                    )
                    raise RuntimeError("Analyzer did not return a result.")
                self._record_ensemble_turn(
                    step_executed=bool(result.step_executed),
                    retryable_failure=bool(result.retryable_failure),
                    yielded_control=bool(getattr(result, "yielded_control", False)),
                    tool_turns_exhausted=False,
                )
""",
        "turn dispatch",
    )
    text = _replace_once(
        text,
        "        finally:\n            total_tokens = _analyzer_reported_tokens(self.analyzer)\n",
        "        finally:\n"
        "            print(\"FLASH_ENSEMBLE_HEALTH\", "
        "json.dumps(self.ensemble_controller.health_summary(), sort_keys=True), flush=True)\n"
        "            total_tokens = _analyzer_reported_tokens(self.analyzer)\n",
        "privacy-safe health record",
    )
    text = _replace_once(
        text,
        "        if flush_viewer_payload:\n            self.write_viewer_payload()\n        return payload\n\n\n@dataclass\nclass HarnessSolver",
        "        if flush_viewer_payload:\n            self.write_viewer_payload()\n"
        "        self.ensemble_last_action_result = dict(payload)\n"
        "        return payload\n\n\n@dataclass\nclass HarnessSolver",
        "action result capture",
    )
    text = _replace_once(
        text,
        """    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }
""",
        """    def _error_payload(self, message: str) -> dict[str, Any]:
        payload = {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }
        self.ensemble_last_action_result = dict(payload)
        return payload
""",
        "error result capture",
    )
    text = _replace_once(
        text,
        """            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
""",
        """            **self.timing_payload(),
        }
        self.ensemble_last_action_result = dict(payload)
        return payload

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
""",
        "terminal result capture",
    )
    text = _replace_once(
        text,
        "    def _finish_if_needed(self) -> None:\n",
        """    def _record_ensemble_turn(
        self,
        *,
        step_executed: bool,
        retryable_failure: bool,
        yielded_control: bool,
        tool_turns_exhausted: bool,
    ) -> None:
        payload = self.ensemble_last_action_result
        reason = self.ensemble_controller.record_turn(
            TurnSignal(
                action_result=payload,
                step_executed=step_executed,
                retryable_failure=retryable_failure,
                yielded_control=yielded_control,
                tool_turns_exhausted=tool_turns_exhausted,
                remaining_seconds=self.timing_payload().get("time_remaining_seconds"),
                current_frame={
                    "ascii": str(self.current_frame().ascii)[:16384],
                    "step": int(self.current_frame().step),
                    "level": int(self.current_frame().level),
                    "shape": list(self.current_frame().shape),
                },
                valid_actions=_engine_action_names(self.game),
                level=_level_number(self.game),
                levels_completed=int(self.game.current_state.levels_completed),
                score=int(self.game.current_state.levels_completed),
                state=str(self.game.current_state.raw.state.name),
            )
        )
        if not self.ensemble_controller.can_handoff(reason):
            return
        snapshot = self.ensemble_controller.handoff(
            reason,
            game_id=self.game.game_run.game_id,
            current_frame={
                "ascii": str(self.current_frame().ascii)[:16384],
                "step": int(self.current_frame().step),
                "level": int(self.current_frame().level),
                "shape": list(self.current_frame().shape),
            },
            valid_actions=_engine_action_names(self.game),
            level=_level_number(self.game),
            levels_completed=int(self.game.current_state.levels_completed),
            score=int(self.game.current_state.levels_completed),
            state=str(self.game.current_state.raw.state.name),
        )
        factory = getattr(self.solver, "hybrid_recovery_factory", None)
        if not callable(factory):
            raise RuntimeError(
                "Hybrid recovery handoff requested but no genuine Hybrid adapter is installed."
            )
        hybrid = factory(self.game, self.game_index)
        if not callable(getattr(hybrid, "re_ground", None)):
            raise RuntimeError("Hybrid adapter must expose re_ground(snapshot).")
        if not callable(getattr(hybrid, "analyze", None)):
            raise RuntimeError("Hybrid adapter must expose analyze(...).")
        hybrid.re_ground(snapshot.as_dict())
        self.analyzer = hybrid

    def _finish_if_needed(self) -> None:
""",
        "controller method",
    )
    text = _replace_once(
        text,
        "        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)\n",
        "        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)\n"
        "        self.ensemble_controller.record_reset_transition(level=_level_number(self.game))\n",
        "automatic reset",
    )
    text = _replace_once(
        text,
        "        if self.stop_event.is_set():\n            return True\n",
        "        if self.stop_event.is_set():\n            return True\n"
        "        if self.ensemble_controller.recovery_exhausted:\n            return True\n",
        "recovery budget stop",
    )
    text = _replace_once(
        text,
        "        self.ensemble_last_action_result = dict(payload)\n        return payload\n\n\n@dataclass\nclass HarnessSolver",
        "        self.ensemble_last_action_result = dict(payload)\n"
        "        self.ensemble_controller.note_recovery_action()\n"
        "        return payload\n\n\n@dataclass\nclass HarnessSolver",
        "recovery action count",
    )
    text = _replace_once(
        text,
        "    analyzer_factory: AnalyzerFactory | None = field(\n"
        "        default=None, repr=False, compare=False\n"
        "    )\n",
        "    analyzer_factory: AnalyzerFactory | None = field(\n"
        "        default=None, repr=False, compare=False\n"
        "    )\n"
        "    hybrid_recovery_factory: AnalyzerFactory | None = field(\n"
        "        default=None, repr=False, compare=False\n"
        "    )\n",
        "hybrid factory field",
    )
    compile(text, "arc_ensemble_solver.py", "exec")
    return text
