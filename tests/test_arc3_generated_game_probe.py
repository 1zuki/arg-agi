from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "arc3_generated_game_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("arc3_generated_game_probe", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = load_module()


class GeneratedGameProbeTests(unittest.TestCase):
    def test_same_seed_is_byte_stable(self):
        left = PROBE.generate_game(17, "switch").as_dict()
        right = PROBE.generate_game(17, "switch").as_dict()
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))

    def test_different_seeds_change_layout(self):
        left = PROBE.generate_game(17, "switch").as_dict()
        right = PROBE.generate_game(18, "switch").as_dict()
        self.assertNotEqual(left, right)

    def test_reset_restores_initial_state(self):
        game = PROBE.generate_game(3, "key")
        state = PROBE.GameState(game)
        initial = state.observe()
        state.step("DOWN")
        self.assertEqual(state.reset(), initial)

    def test_oracle_clears_every_generated_family(self):
        report = PROBE.evaluate(range(5), levels=3)
        self.assertEqual(report["wins"], report["games"])
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_geometry_transforms_preserve_oracle_completions(self):
        for geometry in PROBE.GEOMETRIES:
            with self.subTest(geometry=geometry):
                report = PROBE.evaluate([101], levels=2, geometry=geometry)
                self.assertEqual(report["wins"], report["games"])
                self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_held_out_seed_evaluation_is_explicit(self):
        train = PROBE.evaluate([0, 1, 2])
        held_out = PROBE.evaluate([101, 102])
        self.assertEqual(train["games"], 9)
        self.assertEqual(held_out["games"], 6)
        self.assertEqual(held_out["wins"], held_out["games"])

    def test_games_do_not_share_mutable_state(self):
        left = PROBE.GameState(PROBE.generate_game(4, "direct"))
        right = PROBE.GameState(PROBE.generate_game(4, "direct"))
        left.step("DOWN")
        self.assertNotEqual(left.observe()["steps"], right.observe()["steps"])
        self.assertEqual(right.observe()["position"], [0, 0])

    def test_invalid_action_is_rejected(self):
        state = PROBE.GameState(PROBE.generate_game(0, "direct"))
        with self.assertRaises(ValueError):
            state.step("ACTION7")

    def test_public_observation_hides_generator_fields(self):
        state = PROBE.GameState(PROBE.generate_game(0, "switch"))
        public = state.public_observe()
        self.assertNotIn("game_id", public)
        self.assertNotIn("family", public)
        self.assertNotIn("goal", public)
        self.assertNotIn("special", public)
        self.assertNotIn("hazards", public)
        self.assertNotIn("unlocked", public)
        self.assertIn("grid", public)

    def test_hazard_loss_resets_to_start_with_public_feedback(self):
        game = PROBE.generate_game(101, "hazard", levels=1)
        level = game.levels[0]
        self.assertTrue(level.hazards)
        hazard = next(iter(level.hazards))
        state = PROBE.GameState(game)
        state.position = (hazard[0], hazard[1] - 1)
        result = state.step("RIGHT")
        self.assertTrue(result["life_lost"])
        self.assertTrue(result["reset"])
        self.assertEqual(result["public_state"]["position"], [0, 0])
        self.assertTrue(result["public_state"]["last_action_result"]["life_lost"])

    def test_path_hazard_is_hidden_and_lies_on_nominal_path(self):
        game = PROBE.generate_game(101, "path_hazard", levels=2)
        self.assertTrue(PROBE.GameState(game).unlocked)
        for level in game.levels:
            self.assertTrue(level.hazards)
            self.assertTrue(set(level.hazards).issubset(set(level.path[1:-1])))
            self.assertTrue(
                any(
                    PROBE._has_public_route(
                        level.start,
                        level.goal,
                        width=level.width,
                        height=level.height,
                        walls=set(level.walls),
                        blocked=hazard,
                    )
                    for hazard in level.hazards
                )
            )
        public = PROBE.GameState(game).public_observe()
        self.assertNotIn("hazards", public)

    def test_public_route_accepts_multiple_hidden_blocks(self):
        self.assertTrue(
            PROBE._has_public_route(
                (0, 0),
                (0, 3),
                width=4,
                height=2,
                walls=set(),
                blocked=frozenset({(0, 1), (0, 2)}),
            )
        )

    def test_interior_path_builder_handles_adjacent_goals(self):
        path = PROBE._ensure_interior_path(
            (0, 0),
            (1, 1),
            [(0, 0), (1, 0), (1, 1)],
            width=5,
            height=5,
            rng=random.Random(108),
            minimum_interior=2,
        )
        self.assertGreaterEqual(len(path) - 2, 2)
        self.assertEqual(len(path), len(set(path)))

    def test_blind_policy_wins_on_held_out_seeds(self):
        report = PROBE.evaluate_blind_policy([101, 102], levels=3, max_actions=512)
        self.assertEqual(report["wins"], report["games"])
        self.assertEqual(report["levels_completed"], report["levels_total"])
        self.assertEqual(sum(item["wins"] for item in report["by_family"].values()), report["games"])

    def test_blind_policy_does_not_read_hidden_state(self):
        game = PROBE.generate_game(7, "key")
        state = PROBE.GameState(game)
        policy = PROBE.BlindExplorer()
        for _ in range(32):
            if state.done:
                break
            public = state.public_observe()
            action = policy.choose_action(public)
            self.assertIn(action, PROBE.ACTIONS)
            state.step(action)

    def test_two_phase_policy_starts_with_movement(self):
        game = PROBE.generate_game(7, "direct", levels=1)
        policy = PROBE.TwoPhaseExplorer()
        action = policy.choose_action(PROBE.GameState(game).public_observe())
        self.assertIn(action, ("DOWN", "RIGHT"))

    def test_two_phase_policy_wins_on_held_out_seeds(self):
        report = PROBE.evaluate_two_phase_policy([101, 102], levels=3, max_actions=512)
        self.assertEqual(report["wins"], report["games"])
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_two_phase_policy_does_not_read_hidden_state(self):
        game = PROBE.generate_game(7, "switch")
        state = PROBE.GameState(game)
        policy = PROBE.TwoPhaseExplorer()
        for _ in range(64):
            if state.done:
                break
            public = state.public_observe()
            action = policy.choose_action(public)
            self.assertIn(action, PROBE.ACTIONS)
            state.step(action)

    def test_two_phase_recovery_policy_handles_hidden_hazards(self):
        report = PROBE.evaluate_two_phase_recovery_policy([101, 102], levels=3, max_actions=1024)
        self.assertEqual(report["wins"], report["games"])
        self.assertGreaterEqual(report["life_losses"], 1)
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_public_corridor_policy_starts_with_movement(self):
        game = PROBE.generate_game(7, "switch", levels=1)
        policy = PROBE.PublicCorridorExplorer()
        action = policy.choose_action(PROBE.GameState(game).public_observe())
        self.assertIn(action, ("DOWN", "RIGHT"))

    def test_public_corridor_policy_uses_observed_start_geometry(self):
        level = PROBE.LevelSpec(
            width=3,
            height=3,
            start=(2, 2),
            goal=(0, 0),
            walls=frozenset(),
            special=None,
            path=((2, 2), (2, 1), (2, 0), (1, 0), (0, 0)),
        )
        game = PROBE.GeneratedGame(
            game_id="reverse-corner",
            seed=0,
            family="direct",
            levels=(level,),
        )
        report = PROBE._run_policy(
            game,
            PROBE.PublicCorridorExplorer,
            max_actions=64,
        )
        self.assertTrue(report["won"])

    def test_public_corridor_policy_wins_on_held_out_seeds(self):
        report = PROBE.evaluate_corridor_policy([101, 102], levels=3, max_actions=512)
        self.assertEqual(report["wins"], report["games"])
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_public_corridor_policy_does_not_read_hidden_state(self):
        game = PROBE.generate_game(7, "switch")
        state = PROBE.GameState(game)
        policy = PROBE.PublicCorridorExplorer()
        for _ in range(192):
            if state.done:
                break
            public = state.public_observe()
            action = policy.choose_action(public)
            self.assertIn(action, PROBE.ACTIONS)
            state.step(action)

    def test_public_corridor_policy_handles_hidden_hazards(self):
        report = PROBE.evaluate_corridor_recovery_policy([101, 102], levels=3, max_actions=1024)
        self.assertEqual(report["wins"], report["games"])
        self.assertGreaterEqual(report["life_losses"], 1)
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_policies_handle_alternate_public_geometries(self):
        for geometry in ("canonical", "flip_vertical", "flip_horizontal", "rotate180"):
            with self.subTest(geometry=geometry):
                for evaluate_policy in (
                    PROBE.evaluate_two_phase_policy,
                    PROBE.evaluate_corridor_policy,
                ):
                    report = evaluate_policy(
                        [101, 102],
                        levels=3,
                        max_actions=512,
                        geometry=geometry,
                    )
                    self.assertEqual(report["wins"], report["games"])

    def test_non_far_goal_layouts_are_deterministic_and_oracle_valid(self):
        for geometry in ("near_goal", "offset_goal", "random_goal"):
            with self.subTest(geometry=geometry):
                left = PROBE.generate_game(101, "switch", levels=3, geometry=geometry)
                right = PROBE.generate_game(101, "switch", levels=3, geometry=geometry)
                self.assertEqual(left.as_dict(), right.as_dict())
                for level in left.levels:
                    self.assertEqual(level.path[0], level.start)
                    self.assertEqual(level.path[-1], level.goal)
                    self.assertTrue(all(cell not in level.walls for cell in level.path))
                report = PROBE.evaluate([101, 102], levels=3, geometry=geometry)
                self.assertEqual(report["wins"], report["games"])

    def test_sequence_mechanic_exposes_progress_without_targets(self):
        game = PROBE.generate_game(101, "sequence", levels=1)
        level = game.levels[0]
        state = PROBE.GameState(game)
        public = state.public_observe()
        self.assertIn("sequence_progress", public)
        self.assertNotIn("sequence_points", public)
        self.assertEqual(public["sequence_progress"], 0)
        self.assertTrue(level.sequence_points)
        state.position = level.sequence_points[0]
        result = state.step("TOGGLE")
        self.assertEqual(result["sequence_progress"], 1)
        self.assertEqual(result["public_state"]["sequence_progress"], 1)

    def test_sequence_oracle_and_sequence_policy_complete_held_out_games(self):
        for seed in (101, 102):
            game = PROBE.generate_game(seed, "sequence", levels=3)
            oracle = PROBE.replay(game, PROBE.oracle_actions(game))
            self.assertTrue(oracle["won"])
        report = PROBE.evaluate_sequence_benchmark([101, 102], levels=3, max_actions=1024)
        self.assertEqual(report["policies"]["sequence"]["wins"], 2)
        self.assertEqual(report["policies"]["sequence"]["levels_completed"], 6)

    def test_transformed_sequence_levels_preserve_interaction_points(self):
        for geometry in ("flip_vertical", "flip_horizontal", "rotate180"):
            with self.subTest(geometry=geometry):
                game = PROBE.generate_game(101, "sequence", levels=3, geometry=geometry)
                self.assertTrue(all(level.sequence_points for level in game.levels))
                oracle = PROBE.replay(game, PROBE.oracle_actions(game))
                self.assertTrue(oracle["won"])

    def test_delayed_feedback_hides_timing_and_releases_progress_later(self):
        game = PROBE.generate_game(101, "delayed_sequence", levels=1)
        self.assertEqual(game.as_dict(), PROBE.generate_game(101, "delayed_sequence", levels=1).as_dict())
        level = game.levels[0]
        self.assertGreaterEqual(level.feedback_delay, 1)
        self.assertGreaterEqual(level.feedback_noise_period, 1)
        state = PROBE.GameState(game)
        public = state.public_observe()
        self.assertNotIn("feedback_delay", public)
        self.assertNotIn("feedback_noise_period", public)
        self.assertNotIn("sequence_points", public)
        state.position = level.sequence_points[0]
        state.step("TOGGLE")
        self.assertEqual(state.sequence_progress, 0)
        for _ in range(level.feedback_delay - 1):
            state.step("TOGGLE")
            self.assertEqual(state.sequence_progress, 0)
        state.step("TOGGLE")
        self.assertEqual(state.sequence_progress, 1)

    def test_delayed_feedback_noise_postpones_a_due_update_one_step(self):
        game = PROBE.generate_game(101, "delayed_sequence", levels=1)
        level = game.levels[0]
        state = PROBE.GameState(game)
        state.position = level.sequence_points[0]
        state.step("TOGGLE")
        state.step("TOGGLE")
        state.position = level.sequence_points[1]
        state.step("TOGGLE")
        self.assertEqual(state.steps, 3)
        self.assertEqual(state.sequence_progress, 1)
        state.step("TOGGLE")
        self.assertEqual(state.steps, 4)
        state.step("TOGGLE")
        self.assertEqual(state.steps, 5)
        self.assertEqual(state.sequence_progress, 1)
        state.step("TOGGLE")
        self.assertEqual(state.steps, 6)
        self.assertEqual(state.sequence_progress, 2)

    def test_delayed_feedback_oracle_completes_held_out_games(self):
        for geometry in PROBE.GEOMETRIES:
            with self.subTest(geometry=geometry):
                game = PROBE.generate_game(101, "delayed_sequence", levels=3, geometry=geometry)
                replay = PROBE.replay(game, PROBE.oracle_actions(game))
                self.assertTrue(replay["won"])
                self.assertEqual(replay["levels_completed"], 3)

    def test_delayed_feedback_benchmark_cli_accepts_partial_comparison(self):
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--delayed-feedback-benchmark",
            "--seeds",
            "101",
            "--levels",
            "2",
            "--max-actions",
            "256",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
            exit_code = PROBE.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["family"], "delayed_sequence")
        self.assertIn("sequence", report["policies"])

    def test_path_hazard_benchmark_cli_accepts_partial_comparison(self):
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--path-hazard-benchmark",
            "--seeds",
            "101",
            "--levels",
            "2",
            "--max-actions",
            "512",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
            exit_code = PROBE.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["family"], "path_hazard")
        self.assertEqual(report["policies"]["baseline"]["games"], 1)

    def test_sequence_benchmark_cli_accepts_partial_comparison(self):
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--sequence-benchmark",
            "--seeds",
            "101",
            "--levels",
            "2",
            "--max-actions",
            "512",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
            exit_code = PROBE.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["family"], "sequence")
        self.assertIn("sequence", report["policies"])

    def test_sequence_benchmark_reuses_generator_seeds_for_each_policy(self):
        report = PROBE.evaluate_sequence_benchmark((seed for seed in (101, 102)), levels=1)
        for policy_report in report["policies"].values():
            self.assertEqual(policy_report["games"], 2)

    def test_two_phase_and_adaptive_cover_non_far_goal_layouts(self):
        for geometry in ("near_goal", "offset_goal", "random_goal"):
            with self.subTest(geometry=geometry):
                for evaluate_policy in (
                    PROBE.evaluate_two_phase_policy,
                    PROBE.evaluate_adaptive_policy,
                ):
                    report = evaluate_policy(
                        [101, 102],
                        levels=3,
                        max_actions=512,
                        geometry=geometry,
                    )
                    self.assertEqual(report["wins"], report["games"])

    def test_adaptive_policy_recovers_when_goal_is_not_farthest(self):
        report = PROBE.evaluate_adaptive_policy(
            [101, 102],
            levels=3,
            max_actions=512,
            geometry="center_goal",
        )
        self.assertEqual(report["wins"], report["games"])
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_adaptive_policy_preserves_canonical_corridor_completion(self):
        report = PROBE.evaluate_adaptive_policy(
            [101, 102],
            levels=3,
            max_actions=512,
        )
        self.assertEqual(report["wins"], report["games"])

    def test_adaptive_recovery_policy_handles_hidden_hazards(self):
        report = PROBE.evaluate_adaptive_recovery_policy(
            [101, 102],
            levels=3,
            max_actions=1024,
        )
        self.assertEqual(report["wins"], report["games"])
        self.assertGreaterEqual(report["life_losses"], 1)

    def test_recovery_policy_records_and_avoids_hidden_hazard(self):
        report = PROBE.evaluate_recovery_policy([101, 102], levels=3, max_actions=1024)
        self.assertEqual(report["wins"], report["games"])
        self.assertGreaterEqual(report["life_losses"], 1)
        self.assertEqual(report["levels_completed"], report["levels_total"])

    def test_budget_curve_is_monotone_in_reported_budget(self):
        report = PROBE.summarize_budget_curve([101, 102], [128, 256, 512], levels=3)
        self.assertEqual([item["max_actions"] for item in report["curve"]], [128, 256, 512])
        self.assertTrue(all(item["games"] == 6 for item in report["curve"]))

    def test_partial_budget_curve_is_a_successful_diagnostic(self):
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--budget-curve",
            "--seeds",
            "2000",
            "2001",
            "2002",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
            exit_code = PROBE.main()

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)

    def test_budget_curve_accepts_explicit_action_budgets_and_geometry(self):
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--budget-curve",
            "--budget-values",
            "64",
            "96",
            "128",
            "--levels",
            "2",
            "--adaptive-policy",
            "--geometry",
            "near_goal",
            "--seeds",
            "101",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", stdout):
            exit_code = PROBE.main()

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual([item["max_actions"] for item in report["curve"]], [64, 96, 128])
        self.assertEqual(report["levels"], 2)
        self.assertLess(report["curve"][0]["wins"], report["curve"][0]["games"])
        self.assertEqual(report["curve"][1]["wins"], report["curve"][1]["games"])


if __name__ == "__main__":
    unittest.main()
