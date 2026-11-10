from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "arc3_adaptive_recovery_ablation",
    ROOT / "scripts/arc3_adaptive_recovery_ablation.py",
)
ABLATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ABLATION)


class AdaptiveRecoveryAblationTests(unittest.TestCase):
    def test_small_paired_ablation_is_reproducible_and_route_checked(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
            summary = ABLATION.run_ablation(
                Path(temp_dir) / "ablation",
                geometries=("canonical", "random_goal"),
                seed_splits=((101, 102),),
                levels=2,
                max_actions=512,
                hazard_counts=(1, 2),
            )
            self.assertEqual(summary["games_per_hazard_count"], 4)
            self.assertEqual(summary["levels_checked"], 16)
            self.assertEqual(set(summary["policies"]), {"1", "2"})
            self.assertEqual(
                summary["policies"]["2"]["adaptive"]["games"],
                4,
            )
            self.assertEqual(len(summary["source_reports"]), 4)
            self.assertTrue((Path(temp_dir) / "ablation" / "summary.json").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
