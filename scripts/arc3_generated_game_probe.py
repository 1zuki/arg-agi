"""Bounded CPU-only probe for ARC-AGI-3-style procedural games.

This module is a local harness experiment, not an official ARC-AGI-3
environment and not a leaderboard surrogate.  It supplies a small deterministic
game family so we can test the engineering contracts that remain observable
without the large model: seeded generation, reset semantics, action schemas,
level progression, held-out seeds, and per-game state isolation.

The generated levels are deliberately simple.  Their value is in making the
evaluation boundary explicit before spending a GPU run or generating training
data for fine-tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
from collections import deque
from typing import Any, Iterable, Sequence


FAMILIES = ("direct", "key", "switch")
RECOVERY_FAMILIES = ("hazard", "path_hazard")
PATH_HAZARD_FAMILIES = ("path_hazard",)
SEQUENCE_FAMILIES = ("sequence",)
DELAYED_FEEDBACK_FAMILIES = ("delayed_sequence",)
ALL_FAMILIES = FAMILIES + RECOVERY_FAMILIES + SEQUENCE_FAMILIES + DELAYED_FEEDBACK_FAMILIES
GEOMETRIES = (
    "canonical",
    "flip_vertical",
    "flip_horizontal",
    "rotate180",
    "center_goal",
    "near_goal",
    "offset_goal",
    "random_goal",
)
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "TOGGLE")
_FAMILY_OFFSETS = {
    "direct": 11,
    "key": 23,
    "switch": 37,
    "hazard": 53,
    "sequence": 67,
    "delayed_sequence": 79,
    "path_hazard": 89,
}


Position = tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    width: int
    height: int
    start: Position
    goal: Position
    walls: frozenset[Position]
    special: Position | None
    path: tuple[Position, ...]
    hazards: frozenset[Position] = frozenset()
    sequence_points: tuple[Position, ...] = ()
    feedback_delay: int = 0
    feedback_noise_period: int = 0


@dataclass(frozen=True)
class GeneratedGame:
    game_id: str
    seed: int
    family: str
    levels: tuple[LevelSpec, ...]
    geometry: str = "canonical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "family": self.family,
            "geometry": self.geometry,
            "levels": [
                {
                    "width": level.width,
                    "height": level.height,
                    "start": list(level.start),
                    "goal": list(level.goal),
                    "walls": [list(cell) for cell in sorted(level.walls)],
                    "special": list(level.special) if level.special else None,
                    "path": [list(cell) for cell in level.path],
                    "hazards": [list(cell) for cell in sorted(level.hazards)],
                    "sequence_points": [list(cell) for cell in level.sequence_points],
                    "feedback_delay": level.feedback_delay,
                    "feedback_noise_period": level.feedback_noise_period,
                }
                for level in self.levels
            ],
        }


@dataclass
class GameState:
    game: GeneratedGame
    level_index: int = 0
    position: Position | None = None
    unlocked: bool = False
    steps: int = 0
    done: bool = False
    levels_completed: int = 0
    sequence_progress: int = 0

    def __post_init__(self) -> None:
        self._feedback: dict[str, Any] | None = None
        self._sequence_actual_progress = 0
        self._sequence_feedback_queue: deque[tuple[int, int]] = deque()
        self.reset()

    @property
    def level(self) -> LevelSpec:
        return self.game.levels[self.level_index]

    def reset(self) -> dict[str, Any]:
        self.level_index = 0
        self.levels_completed = 0
        self.done = False
        self.steps = 0
        self._reset_level()
        return self.observe()

    def _reset_level(self) -> None:
        self.position = self.level.start
        self.unlocked = self.game.family in {"direct", "hazard", "path_hazard"}
        self._sequence_actual_progress = 0
        self.sequence_progress = 0
        self._sequence_feedback_queue.clear()
        self._feedback = None

    def _is_sequence_family(self) -> bool:
        return self.game.family in SEQUENCE_FAMILIES + DELAYED_FEEDBACK_FAMILIES

    def _advance_sequence_feedback(self) -> None:
        """Release delayed progress after the action step has been counted."""

        if self.game.family not in DELAYED_FEEDBACK_FAMILIES:
            return
        period = self.level.feedback_noise_period
        while self._sequence_feedback_queue and self._sequence_feedback_queue[0][0] <= self.steps:
            _, progress = self._sequence_feedback_queue.popleft()
            if period and self.steps % period == 0:
                # Deterministic one-step feedback noise.  The actual mechanic
                # state is unchanged; only the public observation is delayed.
                self._sequence_feedback_queue.appendleft((self.steps + 1, progress))
                break
            self.sequence_progress = max(self.sequence_progress, progress)

    def observe(self) -> dict[str, Any]:
        level = self.level
        return {
            "game_id": self.game.game_id,
            "family": self.game.family,
            "level": self.level_index + 1,
            "levels_completed": self.levels_completed,
            "position": list(self.position or level.start),
            "goal": list(level.goal),
            "special": list(level.special) if level.special else None,
            "hazards": [list(cell) for cell in sorted(level.hazards)],
            "unlocked": self.unlocked,
            "sequence_progress": self.sequence_progress,
            "steps": self.steps,
            "done": self.done,
            "available_actions": list(ACTIONS),
            "last_action_result": dict(self._feedback) if self._feedback else None,
        }

    def public_observe(self) -> dict[str, Any]:
        """Return the observation exposed to the blind scripted policy.

        The board reveals geometry and the agent position only.  Goal,
        mechanic family, special cell, and unlock state remain hidden.
        """

        level = self.level
        grid = []
        for row in range(level.height):
            line = []
            for col in range(level.width):
                cell = (row, col)
                if cell in level.walls:
                    line.append("#")
                elif cell == self.position:
                    line.append("A")
                else:
                    line.append(".")
            grid.append("".join(line))
        return {
            "level": self.level_index + 1,
            "levels_completed": self.levels_completed,
            "position": list(self.position or level.start),
            "grid": grid,
            "steps": self.steps,
            "done": self.done,
            # Sequence progress is intentionally observable feedback. The
            # ordered target cells and total count remain hidden.
            "sequence_progress": self.sequence_progress,
            "available_actions": list(ACTIONS),
            "last_action_result": dict(self._feedback) if self._feedback else None,
        }

    def step(self, action: str) -> dict[str, Any]:
        action = str(action).strip().upper()
        if action not in ACTIONS:
            raise ValueError(f"unsupported action: {action!r}")
        if self.done:
            return {"executed": False, "reason": "game_done", "state": self.observe()}

        level = self.level
        before = self.position
        assert before is not None
        if action == "TOGGLE":
            if self.game.family == "switch" and before == level.special:
                self.unlocked = True
            elif self._is_sequence_family():
                expected = (
                    level.sequence_points[self._sequence_actual_progress]
                    if self._sequence_actual_progress < len(level.sequence_points)
                    else None
                )
                if before == expected:
                    self._sequence_actual_progress += 1
                    if self.game.family in DELAYED_FEEDBACK_FAMILIES:
                        due_step = self.steps + 1 + level.feedback_delay
                        self._sequence_feedback_queue.append(
                            (due_step, self._sequence_actual_progress)
                        )
                    else:
                        self.sequence_progress = self._sequence_actual_progress
                    if self._sequence_actual_progress == len(level.sequence_points):
                        self.unlocked = True
        else:
            deltas = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
            delta = deltas[action]
            candidate = (before[0] + delta[0], before[1] + delta[1])
            if (
                0 <= candidate[0] < level.height
                and 0 <= candidate[1] < level.width
                and candidate not in level.walls
            ):
                self.position = candidate
                if candidate in level.hazards:
                    # The hazard is intentionally absent from public_observe().
                    # The only evidence available to a blind policy is the
                    # observable life-loss/reset result below.
                    self.steps += 1
                    self._reset_level()
                    self._feedback = {
                        "action": action,
                        "board_changed": False,
                        "life_lost": True,
                        "reset": True,
                        "reward": 0,
                    }
                    return {
                        "executed": True,
                        "action": action,
                        "board_changed": False,
                        "life_lost": True,
                        "reset": True,
                        "level_completed": False,
                        "state": self.observe(),
                        "public_state": self.public_observe(),
                    }
                if self.game.family == "key" and candidate == level.special:
                    self.unlocked = True

        self.steps += 1
        self._advance_sequence_feedback()
        completed_level = self.position == level.goal and self.unlocked
        self._feedback = {
            "action": action,
            "board_changed": self.position != before,
            "reward": 1 if completed_level else 0,
            "sequence_progress": self.sequence_progress,
        }
        if completed_level:
            self.levels_completed += 1
            if self.level_index + 1 == len(self.game.levels):
                self.done = True
            else:
                self.level_index += 1
                self._reset_level()
        return {
            "executed": True,
            "action": action,
            "board_changed": self.position != before,
            "level_completed": completed_level,
            "sequence_progress": self.sequence_progress,
            "state": self.observe(),
            "public_state": self.public_observe(),
        }


def _path_cells_between(
    start: Position,
    goal: Position,
    rng: random.Random,
) -> list[Position]:
    """Create a randomized axis-aligned path between two cells."""

    row, col = start
    path = [(row, col)]
    row_delta = goal[0] - row
    col_delta = goal[1] - col
    vertical = "DOWN" if row_delta >= 0 else "UP"
    horizontal = "RIGHT" if col_delta >= 0 else "LEFT"
    moves = [vertical] * abs(row_delta) + [horizontal] * abs(col_delta)
    rng.shuffle(moves)
    for move in moves:
        if move == "DOWN":
            row += 1
        elif move == "UP":
            row -= 1
        elif move == "RIGHT":
            col += 1
        else:
            col -= 1
        path.append((row, col))
    assert (row, col) == goal
    return path


def _path_cells(width: int, height: int, rng: random.Random) -> list[Position]:
    """Create a randomized monotone path from the start to the goal."""

    return _path_cells_between((0, 0), (height - 1, width - 1), rng)


def _ensure_interior_path(
    start: Position,
    goal: Position,
    path: list[Position],
    *,
    width: int,
    height: int,
    rng: random.Random,
    minimum_interior: int = 1,
) -> list[Position]:
    """Build a deterministic simple path with enough interior cells."""

    if type(minimum_interior) is not int or minimum_interior < 1:
        raise ValueError("minimum_interior must be a positive integer")
    if len(path) - 2 >= minimum_interior:
        return path
    if minimum_interior == 1:
        # Preserve the original one-cell adjacent-goal fallback so the
        # established one-hazard benchmark remains byte-for-byte comparable.
        neighbors = [
            (start[0] + drow, start[1] + dcol)
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ]
        rng.shuffle(neighbors)
        for detour in neighbors:
            if not (0 <= detour[0] < height and 0 <= detour[1] < width):
                continue
            if detour == goal:
                continue
            queue: deque[Position] = deque([detour])
            parent: dict[Position, Position | None] = {detour: None}
            while queue and goal not in parent:
                row, col = queue.popleft()
                next_cells = [
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ]
                rng.shuffle(next_cells)
                for candidate in next_cells:
                    if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                        continue
                    if candidate == start or candidate in parent:
                        continue
                    parent[candidate] = (row, col)
                    queue.append(candidate)
            if goal in parent:
                suffix: list[Position] = []
                cursor: Position | None = goal
                while cursor is not None:
                    suffix.append(cursor)
                    cursor = parent[cursor]
                suffix.reverse()
                return [start, *suffix]
        raise RuntimeError("path_hazard family could not construct an interior detour")
    # A depth-first search over simple paths is small and deterministic on the
    # 5x5/6x6 diagnostic boards. It avoids returning to the start, which was
    # the source of the earlier adjacent random-goal fixture failure.
    stack: list[tuple[Position, list[Position], set[Position]]] = [
        (start, [start], {start})
    ]
    while stack:
        current, current_path, seen = stack.pop()
        if current == goal and len(current_path) - 2 >= minimum_interior:
            return current_path
        row, col = current
        next_cells = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        rng.shuffle(next_cells)
        for candidate in reversed(next_cells):
            if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                continue
            if candidate in seen:
                continue
            if candidate == goal and len(current_path) - 1 < minimum_interior:
                continue
            stack.append((candidate, [*current_path, candidate], {*seen, candidate}))
    raise RuntimeError("path_hazard family could not construct an interior detour")


def _has_public_route(
    start: Position,
    goal: Position,
    *,
    width: int,
    height: int,
    walls: set[Position],
    blocked: Position | Iterable[Position],
) -> bool:
    """Check that hidden hazards can be avoided on the visible grid.

    A single ``Position`` remains accepted for the original diagnostic.  An
    iterable of positions lets stress tests validate a route around multiple
    hidden hazards without exposing them through the public observation.
    """

    if (
        isinstance(blocked, tuple)
        and len(blocked) == 2
        and all(isinstance(value, int) for value in blocked)
    ):
        blocked_cells = {blocked}
    else:
        blocked_cells = set(blocked)

    queue: deque[Position] = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + drow, col + dcol)
            if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                continue
            if candidate in walls or candidate in blocked_cells or candidate in seen:
                continue
            seen.add(candidate)
            queue.append(candidate)
    return False


def _transform_position(
    position: Position,
    *,
    width: int,
    height: int,
    geometry: str,
) -> Position:
    row, col = position
    if geometry == "canonical":
        return position
    if geometry == "flip_vertical":
        return (height - 1 - row, col)
    if geometry == "flip_horizontal":
        return (row, width - 1 - col)
    if geometry == "rotate180":
        return (height - 1 - row, width - 1 - col)
    raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")


def _transform_level(level: LevelSpec, geometry: str) -> LevelSpec:
    transform = lambda cell: _transform_position(
        cell,
        width=level.width,
        height=level.height,
        geometry=geometry,
    )
    return LevelSpec(
        width=level.width,
        height=level.height,
        start=transform(level.start),
        goal=transform(level.goal),
        walls=frozenset(transform(cell) for cell in level.walls),
        special=transform(level.special) if level.special is not None else None,
        path=tuple(transform(cell) for cell in level.path),
        hazards=frozenset(transform(cell) for cell in level.hazards),
        sequence_points=tuple(transform(cell) for cell in level.sequence_points),
        feedback_delay=level.feedback_delay,
        feedback_noise_period=level.feedback_noise_period,
    )


def _layout_positions(
    width: int,
    height: int,
    level_index: int,
    rng: random.Random,
    geometry: str,
) -> tuple[Position, Position]:
    """Choose deterministic visible start/hidden-goal layouts for a level."""

    if geometry in {"canonical", "flip_vertical", "flip_horizontal", "rotate180", "center_goal"}:
        start = (0, 0)
        goal = (height // 2, width // 2) if geometry == "center_goal" else (height - 1, width - 1)
        return start, goal
    if geometry == "near_goal":
        return (0, 0), (min(1, height - 1), min(1, width - 1))
    if geometry == "offset_goal":
        if level_index % 2 == 0:
            start = (height - 1, 0)
            goal = (1, max(1, width - 2))
        else:
            start = (0, width - 1)
            goal = (max(1, height - 2), 1)
        if goal == start:
            goal = (height // 2, width // 2)
        return start, goal
    if geometry == "random_goal":
        cells = [(row, col) for row in range(height) for col in range(width)]
        start = rng.choice(cells)
        goal = rng.choice([cell for cell in cells if cell != start])
        return start, goal
    raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")


def generate_game(
    seed: int,
    family: str,
    *,
    levels: int = 3,
    geometry: str = "canonical",
) -> GeneratedGame:
    """Generate an independent, reproducible game instance."""

    if family not in ALL_FAMILIES:
        raise ValueError(f"family must be one of {ALL_FAMILIES}, got {family!r}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    if type(levels) is not int or levels < 1:
        raise ValueError("levels must be a positive integer")

    game_rng = random.Random(seed * 1009 + _FAMILY_OFFSETS[family])
    specs: list[LevelSpec] = []
    for level_index in range(levels):
        width = 5 + (level_index % 2)
        height = 5 + ((level_index + 1) % 2)
        start, goal = _layout_positions(width, height, level_index, game_rng, geometry)
        path = _path_cells_between(start, goal, game_rng)
        if family == "path_hazard":
            path = _ensure_interior_path(
                start,
                goal,
                path,
                width=width,
                height=height,
                rng=game_rng,
            )
        path_set = set(path)
        walls = {
            (row, col)
            for row in range(height)
            for col in range(width)
            if (row, col) not in path_set and game_rng.random() < 0.28
        }
        special = None
        if family in {"key", "switch"}:
            special = path[max(1, len(path) // 3)]
        sequence_points: tuple[Position, ...] = ()
        if family in SEQUENCE_FAMILIES + DELAYED_FEEDBACK_FAMILIES:
            # Ordered interactions include the visible start and hidden goal;
            # the middle point exercises a genuinely multi-step state change.
            point_indices = sorted({0, len(path) // 2, len(path) - 1})
            sequence_points = tuple(path[index] for index in point_indices)
        feedback_delay = 0
        feedback_noise_period = 0
        if family in DELAYED_FEEDBACK_FAMILIES:
            feedback_delay = 2 + (level_index % 2)
            feedback_noise_period = 5
        hazards: frozenset[Position] = frozenset()
        if family in RECOVERY_FAMILIES:
            if family == "hazard":
                candidates = [
                    (row, col)
                    for row in range(height)
                    for col in range(width)
                    if (row, col) not in path_set
                    and (row, col) not in walls
                    and (row, col) not in {start, goal}
                ]
                if not candidates:
                    raise RuntimeError("hazard family could not place an off-path hazard")
            else:
                path_candidates = list(path[1:-1])
                if not path_candidates:
                    raise RuntimeError("path_hazard family needs an interior path cell")
                # Keep this diagnostic about recovery rather than impossible
                # geometry: remove deterministic off-path walls until at least
                # one hidden path hazard leaves a public route to the goal.
                while True:
                    candidates = [
                        hazard
                        for hazard in path_candidates
                        if _has_public_route(
                            start,
                            goal,
                            width=width,
                            height=height,
                            walls=walls,
                            blocked=hazard,
                        )
                    ]
                    if candidates:
                        break
                    if not walls:
                        raise RuntimeError(
                            "path_hazard family could not preserve an alternate route"
                        )
                    walls.remove(sorted(walls)[0])
            hazards = frozenset({game_rng.choice(candidates)})
        specs.append(
            LevelSpec(
                width=width,
                height=height,
                start=start,
                goal=goal,
                walls=frozenset(walls),
                special=special,
                path=tuple(path),
                hazards=hazards,
                sequence_points=sequence_points,
                feedback_delay=feedback_delay,
                feedback_noise_period=feedback_noise_period,
            )
        )
    if geometry in {"flip_vertical", "flip_horizontal", "rotate180"}:
        specs = [_transform_level(level, geometry) for level in specs]
    game_id = f"gen-{family}-{seed}"
    if geometry != "canonical":
        game_id = f"gen-{family}-{geometry}-{seed}"
    return GeneratedGame(
        game_id=game_id,
        seed=seed,
        family=family,
        levels=tuple(specs),
        geometry=geometry,
    )


def oracle_actions(game: GeneratedGame) -> list[str]:
    """Return the generator-known path, including the switch interaction."""

    actions: list[str] = []
    for level in game.levels:
        for before, after in zip(level.path, level.path[1:]):
            if game.family == "switch" and before == level.special:
                actions.append("TOGGLE")
            if game.family in SEQUENCE_FAMILIES + DELAYED_FEEDBACK_FAMILIES and before in level.sequence_points:
                actions.append("TOGGLE")
            row_delta = after[0] - before[0]
            col_delta = after[1] - before[1]
            action = _DIRECTION_ACTIONS.get((row_delta, col_delta))
            if action is not None:
                actions.append(action)
            else:  # pragma: no cover - generator only emits adjacent paths
                raise RuntimeError(f"invalid generated path transition: {before} -> {after}")
        if game.family == "switch" and level.path[-1] == level.special:
            actions.append("TOGGLE")
        if game.family in SEQUENCE_FAMILIES + DELAYED_FEEDBACK_FAMILIES and level.path[-1] in level.sequence_points:
            actions.append("TOGGLE")
    return actions


def replay(game: GeneratedGame, actions: Sequence[str]) -> dict[str, Any]:
    state = GameState(game)
    executed = 0
    level_events = 0
    for action in actions:
        result = state.step(action)
        if result["executed"]:
            executed += 1
        if result.get("level_completed"):
            level_events += 1
        if state.done:
            break
    return {
        "game_id": game.game_id,
        "family": game.family,
        "seed": game.seed,
        "won": state.done,
        "levels_completed": state.levels_completed,
        "total_levels": len(game.levels),
        "actions": executed,
        "level_events": level_events,
    }


_DIRECTION_ACTIONS = {
    (-1, 0): "UP",
    (1, 0): "DOWN",
    (0, -1): "LEFT",
    (0, 1): "RIGHT",
}


class BlindExplorer:
    """Explore every visible free cell while probing each cell once.

    The policy never reads hidden generator fields.  It uses only the public
    board geometry, current position, and level number.  A ``TOGGLE`` probe at
    every cell lets it discover the switch family through environment feedback
    without being told where the special cell is.
    """

    def __init__(self) -> None:
        self._level: int | None = None
        self._visited: set[Position] = set()
        self._probed: set[Position] = set()
        self._blocked: set[Position] = set()
        self._pending_position: Position | None = None
        self._pending_action: str | None = None

    @staticmethod
    def _free_cells(observation: dict[str, Any]) -> set[Position]:
        grid = observation.get("grid")
        if not isinstance(grid, list) or not all(isinstance(row, str) for row in grid):
            raise ValueError("public observation must contain a string grid")
        return {
            (row, col)
            for row, line in enumerate(grid)
            for col, value in enumerate(line)
            if value != "#"
        }

    @staticmethod
    def _neighbors(cell: Position, free: set[Position]) -> Iterable[tuple[Position, str]]:
        row, col = cell
        for (drow, dcol), action in _DIRECTION_ACTIONS.items():
            candidate = (row + drow, col + dcol)
            if candidate in free:
                yield candidate, action

    def _reset_if_needed(self, level: int) -> None:
        if level != self._level:
            self._level = level
            self._visited.clear()
            self._probed.clear()
            self._blocked.clear()
            self._pending_position = None
            self._pending_action = None

    def _record_failed_transition(self, observation: dict[str, Any]) -> None:
        result = observation.get("last_action_result")
        if not isinstance(result, dict) or not result.get("life_lost"):
            return
        if self._pending_position is None or self._pending_action is None:
            return
        deltas = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        delta = deltas.get(self._pending_action)
        if delta is not None:
            self._blocked.add(
                (self._pending_position[0] + delta[0], self._pending_position[1] + delta[1])
            )
        self._pending_position = None
        self._pending_action = None

    def _path_to_unvisited(
        self,
        current: Position,
        free: set[Position],
    ) -> list[str] | None:
        queue: deque[Position] = deque([current])
        parent: dict[Position, tuple[Position | None, str | None]] = {
            current: (None, None)
        }
        target: Position | None = None
        while queue:
            cell = queue.popleft()
            if cell not in self._visited:
                target = cell
                break
            for neighbor, action in self._neighbors(cell, free):
                if neighbor not in parent:
                    parent[neighbor] = (cell, action)
                    queue.append(neighbor)
        if target is None:
            return None
        actions: list[str] = []
        cursor = target
        while parent[cursor][0] is not None:
            previous, action = parent[cursor]
            assert previous is not None and action is not None
            actions.append(action)
            cursor = previous
        actions.reverse()
        return actions

    def choose_action(self, observation: dict[str, Any]) -> str:
        """Choose the next action from a public observation only."""

        level = int(observation["level"])
        self._reset_if_needed(level)
        self._record_failed_transition(observation)
        position_data = observation.get("position")
        if not isinstance(position_data, list) or len(position_data) != 2:
            raise ValueError("public observation must contain a two-element position")
        current = (int(position_data[0]), int(position_data[1]))
        free = self._free_cells(observation)
        free.difference_update(self._blocked)
        if current not in free:
            raise ValueError("agent position is not a free cell")
        self._visited.add(current)
        if current not in self._probed:
            self._probed.add(current)
            self._pending_position = current
            self._pending_action = "TOGGLE"
            return "TOGGLE"
        path = self._path_to_unvisited(current, free)
        if path:
            self._pending_position = current
            self._pending_action = path[0]
            return path[0]
        # A complete exploration should have reached the hidden goal.  Keep a
        # deterministic fallback for malformed/generated states.
        self._pending_position = current
        self._pending_action = "TOGGLE"
        return "TOGGLE"


class TwoPhaseExplorer(BlindExplorer):
    """Explore without spending probe actions until movement cannot finish.

    The original explorer probes every newly observed cell.  That is a safe
    diagnostic baseline, but it spends one action per cell even for direct and
    key mechanics where ``TOGGLE`` is irrelevant.  This candidate first walks
    the visible graph, then probes the visited cells only if the movement pass
    did not complete the level, and finally revisits them so a newly discovered
    switch can unlock a goal reached earlier.

    The policy still reads only the public observation.  A life-loss clears the
    current phase and replans around the failed transition, so hidden hazards
    remain a recovery test rather than a generator-field shortcut.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase = "explore"
        self._verify_targets: set[Position] = set()

    def _reset_if_needed(self, level: int) -> None:
        changed = level != self._level
        super()._reset_if_needed(level)
        if changed:
            self._phase = "explore"
            self._verify_targets.clear()

    def _record_failed_transition(self, observation: dict[str, Any]) -> None:
        result = observation.get("last_action_result")
        life_lost = isinstance(result, dict) and bool(result.get("life_lost"))
        super()._record_failed_transition(observation)
        if life_lost:
            self._phase = "explore"
            self._probed.clear()
            self._verify_targets.clear()

    def _path_to_targets(
        self,
        current: Position,
        free: set[Position],
        targets: set[Position],
    ) -> list[str] | None:
        """Return a shortest public-grid path to one of ``targets``."""

        queue: deque[Position] = deque([current])
        parent: dict[Position, tuple[Position | None, str | None]] = {
            current: (None, None)
        }
        target: Position | None = None
        while queue:
            cell = queue.popleft()
            if cell in targets and cell != current:
                target = cell
                break
            for neighbor, action in self._neighbors(cell, free):
                if neighbor not in parent:
                    parent[neighbor] = (cell, action)
                    queue.append(neighbor)
        if target is None:
            return None
        actions: list[str] = []
        cursor = target
        while parent[cursor][0] is not None:
            previous, action = parent[cursor]
            assert previous is not None and action is not None
            actions.append(action)
            cursor = previous
        actions.reverse()
        return actions

    def _set_pending(self, current: Position, action: str) -> str:
        self._pending_position = current
        self._pending_action = action
        return action

    def choose_action(self, observation: dict[str, Any]) -> str:
        """Choose a bounded action from the public observation only."""

        level = int(observation["level"])
        self._reset_if_needed(level)
        self._record_failed_transition(observation)
        position_data = observation.get("position")
        if not isinstance(position_data, list) or len(position_data) != 2:
            raise ValueError("public observation must contain a two-element position")
        current = (int(position_data[0]), int(position_data[1]))
        free = self._free_cells(observation)
        free.difference_update(self._blocked)
        if current not in free:
            raise ValueError("agent position is not a free cell")
        self._visited.add(current)

        if self._phase == "explore":
            path = self._path_to_targets(current, free, free - self._visited)
            if path:
                return self._set_pending(current, path[0])
            self._phase = "toggle"

        if self._phase == "toggle":
            if current not in self._probed:
                self._probed.add(current)
                return self._set_pending(current, "TOGGLE")
            path = self._path_to_targets(current, free, self._visited - self._probed)
            if path:
                return self._set_pending(current, path[0])
            self._phase = "verify"
            self._verify_targets = set(self._visited)

        if self._phase == "verify":
            self._verify_targets.discard(current)
            path = self._path_to_targets(current, free, self._verify_targets)
            if path:
                return self._set_pending(current, path[0])

        # A fully explored and verified board should have completed the level.
        # Keep a deterministic fallback for malformed/generated states.
        return self._set_pending(current, "TOGGLE")


class SequenceExplorer(TwoPhaseExplorer):
    """Use public progress feedback to discover an ordered toggle sequence.

    The ordered target cells and sequence length remain hidden.  The policy
    explores the visible board first, then cycles through visited cells and
    restarts the cycle whenever public sequence progress advances.  This is a
    deliberately small state-tracking probe for observable multi-step games.
    """

    def __init__(self) -> None:
        super().__init__()
        self._interaction_targets: list[Position] = []
        self._interaction_index = 0
        self._observed_sequence_progress = 0

    def _reset_if_needed(self, level: int) -> None:
        changed = level != self._level
        super()._reset_if_needed(level)
        if changed:
            self._interaction_targets.clear()
            self._interaction_index = 0
            self._observed_sequence_progress = 0

    def _record_failed_transition(self, observation: dict[str, Any]) -> None:
        life_lost = bool(
            isinstance(observation.get("last_action_result"), dict)
            and observation["last_action_result"].get("life_lost")
        )
        super()._record_failed_transition(observation)
        if life_lost:
            self._phase = "explore"
            self._interaction_targets.clear()
            self._interaction_index = 0
            self._observed_sequence_progress = 0

    def choose_action(self, observation: dict[str, Any]) -> str:
        level = int(observation["level"])
        self._reset_if_needed(level)
        self._record_failed_transition(observation)
        position_data = observation.get("position")
        if not isinstance(position_data, list) or len(position_data) != 2:
            raise ValueError("public observation must contain a two-element position")
        current = (int(position_data[0]), int(position_data[1]))
        free = self._free_cells(observation)
        free.difference_update(self._blocked)
        if current not in free:
            raise ValueError("agent position is not a free cell")
        self._visited.add(current)

        progress = int(observation.get("sequence_progress", 0))
        if progress != self._observed_sequence_progress:
            self._observed_sequence_progress = progress
            self._interaction_targets = sorted(self._visited)
            self._interaction_index = 0

        if self._phase == "explore":
            path = self._path_to_targets(current, free, free - self._visited)
            if path:
                return self._set_pending(current, path[0])
            self._phase = "interact"
            self._interaction_targets = sorted(self._visited)
            self._interaction_index = 0

        if self._phase == "interact":
            if not self._interaction_targets:
                self._interaction_targets = sorted(self._visited)
                self._interaction_index = 0
            if self._interaction_index >= len(self._interaction_targets):
                self._interaction_index = 0
            target = self._interaction_targets[self._interaction_index]
            if current == target:
                self._interaction_index += 1
                return self._set_pending(current, "TOGGLE")
            path = self._path_to_targets(current, free, {target})
            if path:
                return self._set_pending(current, path[0])
            self._interaction_index += 1
            return self._set_pending(current, "TOGGLE")

        return self._set_pending(current, "TOGGLE")


class PublicCorridorExplorer(BlindExplorer):
    """Follow a public-grid corridor while probing likely mechanic cells.

    The generated layouts expose a useful structural prior: the goal is on a
    geometrically far side of the visible board, and the mechanic cell lies on
    a shortest route to it.  This candidate chooses the free cell with the
    greatest Manhattan distance from the observed start, travels there, then
    toggles every cell in the union of shortest public-grid routes to that
    target.  It never reads the hidden goal, family, special cell, or unlock
    state.  If the target is locked, it returns there after the corridor pass.

    The prior is intentionally explicit.  It is a bounded generated-game
    experiment, not a claim that ARC-AGI-3 games share this geometry.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase = "travel"
        self._target: Position | None = None
        self._corridor: set[Position] = set()
        self._covered: set[Position] = set()
        self._plan: list[str] = []
        self._plan_index = 0

    def _reset_if_needed(self, level: int) -> None:
        changed = level != self._level
        super()._reset_if_needed(level)
        if changed:
            self._phase = "travel"
            self._target = None
            self._corridor.clear()
            self._covered.clear()
            self._plan.clear()
            self._plan_index = 0

    def _record_failed_transition(self, observation: dict[str, Any]) -> None:
        result = observation.get("last_action_result")
        life_lost = isinstance(result, dict) and bool(result.get("life_lost"))
        super()._record_failed_transition(observation)
        if life_lost:
            # The environment resets to the level start. Recompute the public
            # route after excluding the transition that just killed the agent.
            self._phase = "travel"
            self._target = None
            self._corridor.clear()
            self._covered.clear()
            self._plan.clear()
            self._plan_index = 0

    def _bfs_tree(
        self,
        start: Position,
        free: set[Position],
    ) -> tuple[dict[Position, tuple[Position | None, str | None]], dict[Position, int]]:
        queue: deque[Position] = deque([start])
        parent: dict[Position, tuple[Position | None, str | None]] = {
            start: (None, None)
        }
        distance: dict[Position, int] = {start: 0}
        while queue:
            cell = queue.popleft()
            for neighbor, action in self._neighbors(cell, free):
                if neighbor not in parent:
                    parent[neighbor] = (cell, action)
                    distance[neighbor] = distance[cell] + 1
                    queue.append(neighbor)
        return parent, distance

    @staticmethod
    def _actions_from_parent(
        parent: dict[Position, tuple[Position | None, str | None]],
        target: Position,
    ) -> list[str]:
        actions: list[str] = []
        cursor = target
        while parent[cursor][0] is not None:
            previous, action = parent[cursor]
            assert previous is not None and action is not None
            actions.append(action)
            cursor = previous
        actions.reverse()
        return actions

    def _path_to_target(
        self,
        current: Position,
        target: Position,
        free: set[Position],
    ) -> list[str]:
        parent, _ = self._bfs_tree(current, free)
        if target not in parent:
            return []
        return self._actions_from_parent(parent, target)

    def _path_to_any_target(
        self,
        current: Position,
        targets: set[Position],
        free: set[Position],
    ) -> list[str]:
        queue: deque[Position] = deque([current])
        parent: dict[Position, tuple[Position | None, str | None]] = {
            current: (None, None)
        }
        target: Position | None = None
        while queue:
            cell = queue.popleft()
            if cell in targets and cell != current:
                target = cell
                break
            for neighbor, action in self._neighbors(cell, free):
                if neighbor not in parent:
                    parent[neighbor] = (cell, action)
                    queue.append(neighbor)
        if target is None:
            return []
        return self._actions_from_parent(parent, target)

    def _prepare_corridor(self, start: Position, free: set[Position]) -> None:
        parent, distance = self._bfs_tree(start, free)
        if not parent:
            self._target = start
            self._corridor = {start}
            return
        self._target = max(
            parent,
            key=lambda cell: (
                abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
                distance[cell],
                cell[0],
                cell[1],
            ),
        )
        reverse_parent, reverse_distance = self._bfs_tree(self._target, free)
        shortest_length = distance[self._target]
        self._corridor = {
            cell
            for cell in parent
            if cell in reverse_parent
            and distance[cell] + reverse_distance[cell] == shortest_length
        }
        self._covered.clear()
        self._phase = "travel"
        self._plan.clear()
        self._plan_index = 0

    def _emit_plan(self, current: Position) -> str | None:
        if self._plan_index >= len(self._plan):
            self._plan.clear()
            self._plan_index = 0
            return None
        action = self._plan[self._plan_index]
        self._plan_index += 1
        self._pending_position = current
        self._pending_action = action
        return action

    def choose_action(self, observation: dict[str, Any]) -> str:
        """Choose the next bounded action from public observation only."""

        level = int(observation["level"])
        self._reset_if_needed(level)
        self._record_failed_transition(observation)
        position_data = observation.get("position")
        if not isinstance(position_data, list) or len(position_data) != 2:
            raise ValueError("public observation must contain a two-element position")
        current = (int(position_data[0]), int(position_data[1]))
        free = self._free_cells(observation)
        free.difference_update(self._blocked)
        if current not in free:
            raise ValueError("agent position is not a free cell")
        if self._target is None or self._target not in free:
            self._prepare_corridor(current, free)

        planned = self._emit_plan(current)
        if planned is not None:
            return planned

        if self._phase == "travel":
            if current == self._target:
                self._phase = "cover"
            else:
                self._plan = self._path_to_target(current, self._target, free)
                planned = self._emit_plan(current)
                if planned is not None:
                    return planned
                self._phase = "cover"

        if self._phase == "cover":
            self._covered.intersection_update(self._corridor)
            if current in self._corridor and current not in self._covered:
                self._covered.add(current)
                self._pending_position = current
                self._pending_action = "TOGGLE"
                return "TOGGLE"
            remaining = self._corridor - self._covered
            if remaining:
                self._plan = self._path_to_any_target(current, remaining, free)
                planned = self._emit_plan(current)
                if planned is not None:
                    return planned
            self._phase = "return"

        if self._phase == "return":
            if current == self._target:
                self._pending_position = current
                self._pending_action = "TOGGLE"
                return "TOGGLE"
            self._plan = self._path_to_target(current, self._target, free)
            planned = self._emit_plan(current)
            if planned is not None:
                return planned

        # A malformed/disconnected public board gets a deterministic fallback.
        self._pending_position = current
        self._pending_action = "TOGGLE"
        return "TOGGLE"


class AdaptiveCorridorExplorer(BlindExplorer):
    """Use the corridor fast path, then fall back when its prior is contradicted.

    The wrapped corridor policy is allowed to reach its public geometric target
    once.  If the environment has not completed the level there, the policy
    switches to the observation-only two-phase explorer for the remainder of
    that level.  This keeps the measured corridor speed on far-side goals while
    remaining recoverable when a visible board contains a nearer hidden goal.
    """

    def __init__(self) -> None:
        super().__init__()
        self._corridor = PublicCorridorExplorer()
        self._fallback = TwoPhaseExplorer()
        self._level: int | None = None
        self._using_fallback = False

    def choose_action(self, observation: dict[str, Any]) -> str:
        level = int(observation["level"])
        if level != self._level:
            self._level = level
            self._using_fallback = False

        position_data = observation.get("position")
        if not isinstance(position_data, list) or len(position_data) != 2:
            raise ValueError("public observation must contain a two-element position")
        current = (int(position_data[0]), int(position_data[1]))
        if (
            not self._using_fallback
            and self._corridor._phase == "return"
            and self._corridor._target == current
        ):
            self._using_fallback = True

        if self._using_fallback:
            return self._fallback.choose_action(observation)
        return self._corridor.choose_action(observation)


def _run_policy(
    game: GeneratedGame,
    policy_type: type[BlindExplorer],
    *,
    max_actions: int,
) -> dict[str, Any]:
    """Run one observation-only policy with a bounded action budget."""

    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    state = GameState(game)
    policy = policy_type()
    actions = 0
    invalid = 0
    life_losses = 0
    while not state.done and actions < max_actions:
        action = policy.choose_action(state.public_observe())
        try:
            result = state.step(action)
        except ValueError:
            invalid += 1
            break
        actions += 1
        life_losses += int(bool(result.get("life_lost")))
        if not result.get("executed", False):
            break
    return {
        "game_id": game.game_id,
        "family": game.family,
        "seed": game.seed,
        "won": state.done,
        "levels_completed": state.levels_completed,
        "total_levels": len(game.levels),
        "actions": actions,
        "invalid_actions": invalid,
        "life_losses": life_losses,
        "budget_exhausted": not state.done and actions >= max_actions,
    }


def run_blind_policy(game: GeneratedGame, *, max_actions: int = 512) -> dict[str, Any]:
    """Run ``BlindExplorer`` with a bounded action budget."""

    return _run_policy(game, BlindExplorer, max_actions=max_actions)


def run_two_phase_policy(game: GeneratedGame, *, max_actions: int = 512) -> dict[str, Any]:
    """Run the explicit two-phase candidate with a bounded action budget."""

    return _run_policy(game, TwoPhaseExplorer, max_actions=max_actions)


def run_corridor_policy(game: GeneratedGame, *, max_actions: int = 512) -> dict[str, Any]:
    """Run the public-grid corridor candidate with a bounded action budget."""

    return _run_policy(game, PublicCorridorExplorer, max_actions=max_actions)


def run_adaptive_policy(game: GeneratedGame, *, max_actions: int = 512) -> dict[str, Any]:
    """Run the corridor-plus-two-phase fallback policy."""

    return _run_policy(game, AdaptiveCorridorExplorer, max_actions=max_actions)


def run_sequence_policy(game: GeneratedGame, *, max_actions: int = 1024) -> dict[str, Any]:
    """Run the sequence-aware observation-only policy."""

    return _run_policy(game, SequenceExplorer, max_actions=max_actions)


def evaluate(
    seeds: Iterable[int], *, levels: int = 3, geometry: str = "canonical"
) -> dict[str, Any]:
    """Evaluate the deterministic oracle on each family and supplied seed."""

    records = [
        replay(game, oracle_actions(game))
        for seed in seeds
        for family in FAMILIES
        for game in [generate_game(seed, family, levels=levels, geometry=geometry)]
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "records": records,
    }


def evaluate_blind_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 512,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate the observation-only explorer across families and seeds."""

    records = [
        run_blind_policy(game, max_actions=max_actions)
        for seed in seeds
        for family in FAMILIES
        for game in [generate_game(seed, family, levels=levels, geometry=geometry)]
    ]
    by_family = {
        family: {
            "games": sum(record["family"] == family for record in records),
            "wins": sum(record["family"] == family and record["won"] for record in records),
            "mean_actions": round(
                sum(record["actions"] for record in records if record["family"] == family)
                / max(1, sum(record["family"] == family for record in records)),
                3,
            ),
        }
        for family in FAMILIES
    }
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "by_family": by_family,
        "records": records,
    }


def evaluate_recovery_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate recovery from hidden hazards on held-out generated layouts."""

    records = [
        run_blind_policy(
            generate_game(seed, "hazard", levels=levels, geometry=geometry),
            max_actions=max_actions,
        )
        for seed in seeds
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "life_losses": sum(record["life_losses"] for record in records),
        "records": records,
    }


def _evaluate_policy(
    seeds: Iterable[int],
    policy_type: type[BlindExplorer],
    *,
    levels: int,
    max_actions: int,
    geometry: str = "canonical",
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    records = [
        _run_policy(
            generate_game(seed, family, levels=levels, geometry=geometry),
            policy_type,
            max_actions=max_actions,
        )
        for seed in seeds
        for family in families
    ]
    by_family = {
        family: {
            "games": sum(record["family"] == family for record in records),
            "wins": sum(record["family"] == family and record["won"] for record in records),
            "mean_actions": round(
                sum(record["actions"] for record in records if record["family"] == family)
                / max(1, sum(record["family"] == family for record in records)),
                3,
            ),
        }
        for family in families
    }
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "by_family": by_family,
        "records": records,
    }


def evaluate_two_phase_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 512,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate the two-phase candidate across the public mechanic families."""

    return _evaluate_policy(
        seeds,
        TwoPhaseExplorer,
        levels=levels,
        max_actions=max_actions,
        geometry=geometry,
    )


def evaluate_two_phase_recovery_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate two-phase recovery on held-out hidden-hazard layouts."""

    records = [
        run_two_phase_policy(
            generate_game(seed, "hazard", levels=levels, geometry=geometry),
            max_actions=max_actions,
        )
        for seed in seeds
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "life_losses": sum(record["life_losses"] for record in records),
        "records": records,
    }


def evaluate_corridor_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 512,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate the public-grid corridor candidate across mechanic families."""

    return _evaluate_policy(
        seeds,
        PublicCorridorExplorer,
        levels=levels,
        max_actions=max_actions,
        geometry=geometry,
    )


def evaluate_corridor_recovery_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate corridor replanning on held-out hidden-hazard layouts."""

    records = [
        run_corridor_policy(
            generate_game(seed, "hazard", levels=levels, geometry=geometry),
            max_actions=max_actions,
        )
        for seed in seeds
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "life_losses": sum(record["life_losses"] for record in records),
        "records": records,
    }


def evaluate_adaptive_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 512,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate the adaptive corridor fallback across mechanic families."""

    return _evaluate_policy(
        seeds,
        AdaptiveCorridorExplorer,
        levels=levels,
        max_actions=max_actions,
        geometry=geometry,
    )


def evaluate_adaptive_recovery_policy(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Evaluate adaptive recovery on hidden-hazard layouts."""

    records = [
        run_adaptive_policy(
            generate_game(seed, "hazard", levels=levels, geometry=geometry),
            max_actions=max_actions,
        )
        for seed in seeds
    ]
    return {
        "games": len(records),
        "wins": sum(record["won"] for record in records),
        "win_rate": round(sum(record["won"] for record in records) / max(1, len(records)), 3),
        "levels_completed": sum(record["levels_completed"] for record in records),
        "levels_total": sum(record["total_levels"] for record in records),
        "life_losses": sum(record["life_losses"] for record in records),
        "records": records,
    }


def evaluate_sequence_benchmark(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Compare all observation-only policies on the sequence family."""

    seed_list = list(seeds)
    policies: dict[str, type[BlindExplorer]] = {
        "baseline": BlindExplorer,
        "two_phase": TwoPhaseExplorer,
        "corridor": PublicCorridorExplorer,
        "adaptive": AdaptiveCorridorExplorer,
        "sequence": SequenceExplorer,
    }
    return {
        "family": "sequence",
        "geometry": geometry,
        "action_budget": max_actions,
        "policies": {
            name: _evaluate_policy(
                seed_list,
                policy_type,
                levels=levels,
                max_actions=max_actions,
                geometry=geometry,
                families=SEQUENCE_FAMILIES,
            )
            for name, policy_type in policies.items()
        },
    }


def evaluate_delayed_feedback_benchmark(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1536,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Compare the same observation-only policies under delayed feedback."""

    seed_list = list(seeds)
    policies: dict[str, type[BlindExplorer]] = {
        "baseline": BlindExplorer,
        "two_phase": TwoPhaseExplorer,
        "corridor": PublicCorridorExplorer,
        "adaptive": AdaptiveCorridorExplorer,
        "sequence": SequenceExplorer,
    }
    return {
        "family": "delayed_sequence",
        "geometry": geometry,
        "action_budget": max_actions,
        "policies": {
            name: _evaluate_policy(
                seed_list,
                policy_type,
                levels=levels,
                max_actions=max_actions,
                geometry=geometry,
                families=DELAYED_FEEDBACK_FAMILIES,
            )
            for name, policy_type in policies.items()
        },
    }


def evaluate_path_hazard_benchmark(
    seeds: Iterable[int],
    *,
    levels: int = 3,
    max_actions: int = 1024,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Compare recovery policies when a hidden hazard blocks the nominal path."""

    seed_list = list(seeds)
    policies: dict[str, type[BlindExplorer]] = {
        "baseline": BlindExplorer,
        "two_phase": TwoPhaseExplorer,
        "corridor": PublicCorridorExplorer,
        "adaptive": AdaptiveCorridorExplorer,
    }
    return {
        "family": "path_hazard",
        "geometry": geometry,
        "action_budget": max_actions,
        "policies": {
            name: _evaluate_policy(
                seed_list,
                policy_type,
                levels=levels,
                max_actions=max_actions,
                geometry=geometry,
                families=PATH_HAZARD_FAMILIES,
            )
            for name, policy_type in policies.items()
        },
    }


def summarize_budget_curve(
    seeds: Iterable[int],
    budgets: Sequence[int],
    *,
    levels: int = 5,
    policy_type: type[BlindExplorer] = BlindExplorer,
    geometry: str = "canonical",
) -> dict[str, Any]:
    """Return win/level curves for fixed action budgets."""

    seed_list = list(seeds)
    curve: list[dict[str, Any]] = []
    for budget in budgets:
        report = _evaluate_policy(
            seed_list,
            policy_type,
            levels=levels,
            max_actions=budget,
            geometry=geometry,
        )
        curve.append(
            {
                "max_actions": budget,
                "games": report["games"],
                "wins": report["wins"],
                "win_rate": report["win_rate"],
                "levels_completed": report["levels_completed"],
                "levels_total": report["levels_total"],
                "by_family": report["by_family"],
            }
        )
    return {"levels": levels, "curve": curve}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--geometry", choices=GEOMETRIES, default="canonical")
    parser.add_argument("--blind-policy", action="store_true")
    parser.add_argument("--two-phase-policy", action="store_true")
    parser.add_argument("--two-phase-recovery-policy", action="store_true")
    parser.add_argument("--corridor-policy", action="store_true")
    parser.add_argument("--corridor-recovery-policy", action="store_true")
    parser.add_argument("--adaptive-policy", action="store_true")
    parser.add_argument("--adaptive-recovery-policy", action="store_true")
    parser.add_argument("--recovery-policy", action="store_true")
    parser.add_argument("--sequence-benchmark", action="store_true")
    parser.add_argument("--delayed-feedback-benchmark", action="store_true")
    parser.add_argument("--path-hazard-benchmark", action="store_true")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--budget-curve", action="store_true")
    parser.add_argument(
        "--budget-values",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Action budgets for --budget-curve (default: 128 256 512 1024).",
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if any(type(value) is not int or value < 1 for value in args.budget_values):
        parser.error("--budget-values must contain only positive integers.")
    if args.budget_curve:
        result = summarize_budget_curve(
            args.seeds,
            args.budget_values,
            levels=args.levels,
            geometry=args.geometry,
            policy_type=(
                AdaptiveCorridorExplorer
                if args.adaptive_policy
                else PublicCorridorExplorer
                if args.corridor_policy
                else TwoPhaseExplorer
                if args.two_phase_policy
                else BlindExplorer
            ),
        )
    elif args.sequence_benchmark:
        result = evaluate_sequence_benchmark(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.delayed_feedback_benchmark:
        result = evaluate_delayed_feedback_benchmark(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.path_hazard_benchmark:
        result = evaluate_path_hazard_benchmark(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.adaptive_recovery_policy:
        result = evaluate_adaptive_recovery_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.corridor_recovery_policy:
        result = evaluate_corridor_recovery_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.two_phase_recovery_policy:
        result = evaluate_two_phase_recovery_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.recovery_policy:
        result = evaluate_recovery_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.two_phase_policy:
        result = evaluate_two_phase_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.corridor_policy:
        result = evaluate_corridor_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.adaptive_policy:
        result = evaluate_adaptive_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    elif args.blind_policy:
        result = evaluate_blind_policy(
            args.seeds,
            levels=args.levels,
            max_actions=args.max_actions,
            geometry=args.geometry,
        )
    else:
        result = evaluate(args.seeds, levels=args.levels, geometry=args.geometry)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    if args.sequence_benchmark or args.delayed_feedback_benchmark or args.path_hazard_benchmark:
        # Partial policy wins are expected diagnostic data in a comparison.
        return 0
    if args.budget_curve:
        # Partial wins at deliberately small action budgets are expected data,
        # not a failed probe. The JSON report carries the measured result.
        return 0
    return int(result["wins"] != result["games"])


if __name__ == "__main__":
    raise SystemExit(main())
