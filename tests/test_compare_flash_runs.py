from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_flash_runs", ROOT / "scripts" / "compare_flash_runs.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareFlashRunsTests(unittest.TestCase):
    def make_run(self, root: Path, scores: list[float], *, states: list[str] | None = None) -> None:
        ids = [f"game-{i}" for i in range(len(scores))]
        states = states or ["gave_up"] * len(scores)
        root.mkdir()
        (root / "score.json").write_text(json.dumps({
            "score": sum(scores) / len(scores),
            "games": {game_id: {"score": score} for game_id, score in zip(ids, scores)},
            "metadata": {"game_ids": ids},
        }))
        (root / "benchmark.json").write_text(json.dumps({
            "game_runs": [
                {
                    "game_id": game_id,
                    "state": state,
                    "final_score": score,
                    "history": [{}] * (index + 1),
                }
                for index, (game_id, state, score) in enumerate(zip(ids, states, scores))
            ]
        }))

    def test_compare_reports_paired_deltas_and_variance(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            candidate = Path(temp) / "candidate"
            self.make_run(base, [1.0, 2.0, 3.0])
            self.make_run(candidate, [2.0, 1.0, 3.0])
            report = MODULE.compare_runs(base, candidate)
            self.assertEqual(report["wins"], 1)
            self.assertEqual(report["ties"], 1)
            self.assertEqual(report["losses"], 1)
            self.assertEqual(report["worst_regression"], -1.0)
            self.assertEqual(report["mean_delta"], 0.0)
            self.assertEqual(report["per_game"][0]["candidate_actions"], 1)

    def test_incomplete_run_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            self.make_run(root, [1.0], states=["cancelled"])
            with self.assertRaisesRegex(ValueError, "incomplete game"):
                MODULE.load_run(root)

    def test_mismatched_game_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base"
            candidate = Path(temp) / "candidate"
            self.make_run(base, [1.0, 2.0])
            self.make_run(candidate, [1.0, 2.0])
            score = json.loads((candidate / "score.json").read_text())
            score["metadata"]["game_ids"] = ["game-1", "game-0"]
            (candidate / "score.json").write_text(json.dumps(score))
            with self.assertRaisesRegex(ValueError, "metadata game order"):
                MODULE.compare_runs(base, candidate)

    def test_declared_mean_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            self.make_run(root, [1.0, 2.0])
            score = json.loads((root / "score.json").read_text())
            score["score"] = 99
            (root / "score.json").write_text(json.dumps(score))
            with self.assertRaisesRegex(ValueError, "declared score"):
                MODULE.load_run(root)


if __name__ == "__main__":
    unittest.main()
