"""Conservative Flash-primary/Hybrid-recovery controller.

The controller is independent of ARC runtime classes. Runtime adapters provide
normalized turn signals and receive a compact handoff snapshot; they remain
responsible for model calls and environment actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from itertools import islice
from typing import Any, Mapping, Sequence


MAX_HANDOFF_FRAME_CHARS = 16_384
MAX_HANDOFF_STRING_CHARS = 4_096
MAX_HANDOFF_ITEMS = 256
_PRIVATE_KEYS = frozenset(
    {"prompt", "transcript", "reasoning", "raw_context", "stdout", "stderr"}
)


class AgentOwner(str, Enum):
    FLASH = "flash"
    HYBRID = "hybrid"


class HandoffReason(str, Enum):
    REPEAT_NO_CHANGE_GUARD = "repeat_no_change_guard"
    INVALID_ACTION_FAILURE = "invalid_action_failure"
    STALE_TERMINAL_STATE = "stale_terminal_state"
    TOOL_LOOP_EXHAUSTION = "tool_loop_exhaustion"
    TIMEOUT_PRESSURE = "timeout_pressure"


@dataclass(frozen=True)
class ControllerConfig:
    no_progress_window: int = 3
    invalid_action_window: int = 2
    stale_terminal_window: int = 2
    tool_exhaustion_window: int = 2
    max_recovery_actions: int = 12
    max_recovery_turns: int = 16
    timeout_pressure_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "no_progress_window",
            "invalid_action_window",
            "stale_terminal_window",
            "tool_exhaustion_window",
            "max_recovery_actions",
            "max_recovery_turns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.timeout_pressure_seconds) or self.timeout_pressure_seconds < 0:
            raise ValueError("timeout_pressure_seconds must be non-negative")


@dataclass(frozen=True)
class TurnSignal:
    """Runtime-owned evidence from one analyzer turn."""

    result: Any = None
    action_result: Mapping[str, Any] | None = None
    step_executed: bool = False
    yielded_control: bool = False
    retryable_failure: bool = False
    tool_turns_exhausted: bool = False
    remaining_seconds: float | None = None
    zero_tokens: bool = False
    current_frame: Any = None
    valid_actions: Sequence[str] | None = None
    level: int | None = None
    levels_completed: int | None = None
    score: float | int | None = None
    state: str | None = None


@dataclass(frozen=True)
class RecoverySnapshot:
    """Only normalized, bounded state is allowed across a handoff."""

    game_id: str
    level: int | None
    levels_completed: int | None
    score: float | int | None
    state: str | None
    current_frame: Any
    valid_actions: tuple[str, ...]
    last_action: str | None
    last_action_result: dict[str, Any] | None
    no_progress_streak: int
    handoff_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level": self.level,
            "levels_completed": self.levels_completed,
            "score": self.score,
            "state": self.state,
            "current_frame": self.current_frame,
            "valid_actions": list(self.valid_actions),
            "last_action": self.last_action,
            "last_action_result": (
                dict(self.last_action_result)
                if self.last_action_result is not None
                else None
            ),
            "no_progress_streak": self.no_progress_streak,
            "handoff_reason": self.handoff_reason,
        }


@dataclass
class _Memory:
    game_id: str | None = None
    level: int | None = None
    current_frame: Any = None
    valid_actions: tuple[str, ...] = ()
    last_action: str | None = None
    last_action_result: dict[str, Any] | None = None
    recent_results: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=8))
    mechanic_facts: deque[str] = field(default_factory=lambda: deque(maxlen=16))
    no_progress_streak: int = 0
    max_no_progress_streak: int = 0
    invalid_action_count: int = 0
    tool_no_action_streak: int = 0
    zero_token_streak: int = 0
    levels_completed: int | None = None
    score: float | int | None = None
    state: str | None = None
    fresh_observation_required: bool = False

    def reset(self, game_id: str) -> None:
        self.__dict__.update(_Memory(game_id=game_id).__dict__)

    def reset_level(self, level: int | None) -> None:
        self.level = level
        self.current_frame = None
        self.valid_actions = ()
        self.last_action = None
        self.last_action_result = None
        self.recent_results.clear()
        self.no_progress_streak = 0
        self.fresh_observation_required = True


def _valid_action_tuple(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if not isinstance(values, (list, tuple)):
        return None
    return tuple(
        str(value).strip()[:256]
        for value in islice(values, MAX_HANDOFF_ITEMS)
        if str(value).strip()
    )


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-safe, bounded observation for the recovery boundary."""
    if depth > 4:
        return "[depth-limited]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_HANDOFF_STRING_CHARS]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_HANDOFF_ITEMS:
                break
            name = str(key).strip()
            if name.lower() in _PRIVATE_KEYS:
                continue
            result[name[:128]] = _bounded_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth=depth + 1) for item in islice(value, MAX_HANDOFF_ITEMS)]
    return None


def _handoff_frame(value: Any) -> Any:
    bounded = _bounded_value(value)
    if bounded is None:
        return None
    try:
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    if len(encoded) > MAX_HANDOFF_FRAME_CHARS:
        return None
    return bounded


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _false_flags(result: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return all(key in result and result[key] is False for key in keys)


class BoundedRecoveryController:
    """Selects at most one one-way Flash -> Hybrid handoff per game."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.memory = _Memory()
        self.owner = AgentOwner.FLASH
        self.handoff_count = 0
        self.handoff_reason: HandoffReason | None = None
        self.recovery_actions = 0
        self.recovery_turns = 0

    def reset_game(self, game_id: str) -> None:
        self.memory.reset(str(game_id))
        self.owner = AgentOwner.FLASH
        self.handoff_count = 0
        self.handoff_reason = None
        self.recovery_actions = 0
        self.recovery_turns = 0

    @property
    def recovery_exhausted(self) -> bool:
        return self.owner is AgentOwner.HYBRID and (
            self.recovery_actions >= self.config.max_recovery_actions
            or self.recovery_turns >= self.config.max_recovery_turns
        )

    @property
    def no_progress_streak(self) -> int:
        return self.memory.no_progress_streak

    @property
    def fresh_observation_required(self) -> bool:
        return self.memory.fresh_observation_required

    def record_mechanic_fact(self, fact: str) -> None:
        text = str(fact).strip()[:512]
        if text and text not in self.memory.mechanic_facts:
            self.memory.mechanic_facts.append(text)

    def record_reset_transition(self, *, level: int | None = None) -> None:
        """Forget frame-local state after an environment reset before re-grounding."""
        self.memory.reset_level(level)

    def record_turn(self, signal: TurnSignal) -> HandoffReason | None:
        self._compact_memory(signal)
        action_executed = signal.step_executed or (
            isinstance(signal.action_result, Mapping)
            and signal.action_result.get("executed") is True
        )
        if action_executed:
            self.memory.tool_no_action_streak = 0
        elif signal.yielded_control or signal.retryable_failure or signal.tool_turns_exhausted:
            self.memory.tool_no_action_streak += 1
        if self.owner is AgentOwner.HYBRID:
            self.recovery_turns += 1
            return None
        return self._select_reason(signal)

    def note_recovery_action(self) -> None:
        if self.owner is AgentOwner.HYBRID and not self.recovery_exhausted:
            self.recovery_actions += 1

    def can_handoff(self, reason: HandoffReason | None) -> bool:
        return (
            self.owner is AgentOwner.FLASH
            and self.handoff_count == 0
            and reason is not None
            and self.config.max_recovery_actions > 0
        )

    def handoff(
        self,
        reason: HandoffReason,
        *,
        game_id: str,
        current_frame: Any,
        valid_actions: Sequence[str] | None,
        level: int | None,
        levels_completed: int | None,
        score: float | int | None,
        state: str | None,
    ) -> RecoverySnapshot:
        if not self.can_handoff(reason):
            raise RuntimeError("handoff is not permitted")
        if self.memory.fresh_observation_required:
            raise RuntimeError("handoff requires a fresh post-transition observation")
        normalized_frame = _handoff_frame(current_frame)
        if normalized_frame is None:
            raise RuntimeError("handoff requires a bounded current observation")
        normalized_actions = _valid_action_tuple(valid_actions)
        if normalized_actions is None:
            normalized_actions = self.memory.valid_actions
        self.owner = AgentOwner.HYBRID
        self.handoff_count = 1
        self.handoff_reason = reason
        self.memory.current_frame = normalized_frame
        self.memory.valid_actions = normalized_actions
        self.memory.level = level
        self.memory.levels_completed = levels_completed
        self.memory.score = score
        self.memory.state = state
        return RecoverySnapshot(
            game_id=str(game_id),
            level=level,
            levels_completed=levels_completed,
            score=score,
            state=state,
            current_frame=normalized_frame,
            valid_actions=normalized_actions,
            last_action=self.memory.last_action,
            last_action_result=(
                dict(self.memory.last_action_result)
                if self.memory.last_action_result is not None
                else None
            ),
            no_progress_streak=self.memory.no_progress_streak,
            handoff_reason=reason.value,
        )

    def _compact_memory(self, signal: TurnSignal) -> None:
        if signal.level is not None and self.memory.level is not None and signal.level != self.memory.level:
            self.memory.reset_level(signal.level)
        elif signal.level is not None:
            self.memory.level = signal.level
        normalized_frame = _handoff_frame(signal.current_frame)
        if normalized_frame is not None:
            self.memory.current_frame = normalized_frame
            self.memory.fresh_observation_required = False
        action_tuple = _valid_action_tuple(signal.valid_actions)
        if action_tuple is not None:
            self.memory.valid_actions = action_tuple
        self.memory.score = signal.score
        if signal.levels_completed is not None:
            self.memory.levels_completed = signal.levels_completed
        self.memory.state = signal.state
        if signal.zero_tokens:
            self.memory.zero_token_streak += 1
        else:
            self.memory.zero_token_streak = 0
        result = signal.action_result
        if not isinstance(result, Mapping):
            return
        compact = {
            key: result[key]
            for key in (
                "executed", "action_num", "level", "score", "reward", "state",
                "valid_actions", "board_changed", "done", "level_completed",
                "game_over", "run_complete", "action_display", "requested_count",
                "executed_count", "stop_reason",
            )
            if key in result
        }
        if self._is_invalid(result):
            compact["invalid_action"] = True
        self.memory.recent_results.append(compact)
        self.memory.last_action_result = dict(compact)
        action = compact.get("action_display")
        if action is not None:
            self.memory.last_action = str(action)
        if self._confirmed_no_progress(compact):
            self.memory.no_progress_streak += 1
            self.memory.max_no_progress_streak = max(
                self.memory.max_no_progress_streak, self.memory.no_progress_streak
            )
        elif compact.get("executed") is True:
            self.memory.no_progress_streak = 0
        if self._is_invalid(compact):
            self.memory.invalid_action_count += 1

    def _select_reason(self, signal: TurnSignal) -> HandoffReason | None:
        if self._repeat_no_change_ready():
            return HandoffReason.REPEAT_NO_CHANGE_GUARD
        if self._invalid_action_ready():
            return HandoffReason.INVALID_ACTION_FAILURE
        if self._stale_terminal_ready():
            return HandoffReason.STALE_TERMINAL_STATE
        if self._tool_exhaustion_ready(signal):
            return HandoffReason.TOOL_LOOP_EXHAUSTION
        if self._timeout_pressure_ready(signal):
            return HandoffReason.TIMEOUT_PRESSURE
        return None

    def _repeat_no_change_ready(self) -> bool:
        items = list(self.memory.recent_results)[-self.config.no_progress_window :]
        if len(items) != self.config.no_progress_window or not all(
            self._confirmed_no_progress(item) for item in items
        ):
            return False
        actions = {item.get("action_display") for item in items}
        valid_sets = {tuple(item.get("valid_actions", ())) for item in items}
        return len(actions) == 1 and len(valid_sets) == 1

    def _invalid_action_ready(self) -> bool:
        items = list(self.memory.recent_results)[-self.config.invalid_action_window :]
        return len(items) == self.config.invalid_action_window and all(
            self._is_invalid(item) for item in items
        )

    def _stale_terminal_ready(self) -> bool:
        items = list(self.memory.recent_results)[-self.config.stale_terminal_window :]
        if len(items) != self.config.stale_terminal_window:
            return False
        return all(
            item.get("executed") is False
            and isinstance(item.get("valid_actions"), list)
            and any(item.get(key) is True for key in ("done", "game_over", "run_complete"))
            for item in items
        )

    def _tool_exhaustion_ready(self, signal: TurnSignal) -> bool:
        if not (
            signal.yielded_control
            or signal.retryable_failure
            or signal.tool_turns_exhausted
        ):
            return False
        return self.memory.tool_no_action_streak >= self.config.tool_exhaustion_window

    def _timeout_pressure_ready(self, signal: TurnSignal) -> bool:
        remaining = signal.remaining_seconds
        if not _number(remaining) or float(remaining) > self.config.timeout_pressure_seconds:
            return False
        recent = list(self.memory.recent_results)[-self.config.no_progress_window :]
        return len(recent) == self.config.no_progress_window and all(
            self._confirmed_no_progress(item) for item in recent
        )

    @staticmethod
    def _confirmed_no_progress(result: Mapping[str, Any]) -> bool:
        return (
            result.get("executed") is True
            and result.get("requested_count") == 1
            and result.get("executed_count") == 1
            and result.get("board_changed") is False
            and _number(result.get("reward"))
            and float(result["reward"]) == 0.0
            and _false_flags(result, ("done", "level_completed", "game_over", "run_complete"))
            and isinstance(result.get("valid_actions"), list)
            and isinstance(result.get("action_display"), str)
        )

    @staticmethod
    def _is_invalid(result: Mapping[str, Any]) -> bool:
        if result.get("invalid_action") is True:
            return True
        if result.get("executed") is not False or not isinstance(result.get("valid_actions"), list):
            return False
        reason = str(result.get("stop_reason", "")).lower()
        error = str(result.get("error", "")).lower()
        return reason == "invalid_action" or any(
            word in error for word in ("not valid", "unavailable", "rejected")
        )

    def health_summary(self) -> dict[str, Any]:
        """Return aggregate diagnostics suitable for submission-mode logging."""
        return {
            "agent_owner": self.owner.value,
            "hybrid_handoffs": self.handoff_count,
            "handoff_reason": self.handoff_reason.value if self.handoff_reason else None,
            "recovery_actions": self.recovery_actions,
            "recovery_turns": self.recovery_turns,
            "invalid_action_count": self.memory.invalid_action_count,
            "tool_no_action_streak": self.memory.tool_no_action_streak,
            "max_no_progress_streak": self.memory.max_no_progress_streak,
            "zero_token_streak": self.memory.zero_token_streak,
            "recovery_exhausted": self.recovery_exhausted,
        }
