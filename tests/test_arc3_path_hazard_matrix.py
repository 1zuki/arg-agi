from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "arc3_path_hazard_matrix",
    ROOT / "scripts/arc3_path_hazard_matrix.py",
)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class PathHazardMatrixTests(unittest.TestCase):
    def test_small_matrix_writes_reports_and_route_invariant(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
            summary = MATRIX.run_matrix(
                Path(temp_dir) / "path_hazard",
                geometries=("canonical",),
                seed_splits=((101, 102),),
                levels=1,
                max_actions=256,
            )
            self.assertEqual(summary["games"], 2)
            self.assertEqual(summary["levels_total"], 2)
            self.assertEqual(summary["route_invariant"]["invalid_levels"], 0)
            self.assertEqual(summary["policies"]["adaptive"]["games"], 2)
            self.assertTrue((Path(temp_dir) / "path_hazard" / "summary.json").is_file())
            self.assertTrue(
                (Path(temp_dir) / "path_hazard" / "canonical_101_102.json").is_file()
            )

    def test_split_parser_requires_ordered_range(self) -> None:
        self.assertEqual(MATRIX._parse_split("100-199"), (100, 199))
        with self.assertRaises(ValueError):
            MATRIX._parse_split("199-100")
        with self.assertRaises(ValueError):
            MATRIX._parse_split("100")

    def test_multi_hazard_matrix_preserves_route_around_hazard_set(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
            output_dir = Path(temp_dir) / "multi_path_hazard"
            summary = MATRIX.run_matrix(
                output_dir,
                geometries=("canonical", "random_goal"),
                seed_splits=((101, 102),),
                levels=2,
                max_actions=512,
                hazard_count=2,
            )
            self.assertEqual(summary["mechanic"], "multi_path_hazard")
            self.assertEqual(summary["hazard_count"], 2)
            self.assertEqual(summary["games"], 4)
            self.assertEqual(summary["levels_total"], 8)
            self.assertEqual(summary["route_invariant"]["invalid_levels"], 0)
            self.assertEqual(summary["policies"]["adaptive"]["games"], 4)
            report = (output_dir / "random_goal_101_102.json").read_text()
            self.assertIn('"hazard_count": 2', report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
