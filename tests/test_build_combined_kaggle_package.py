from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_combined_kaggle_package",
    ROOT / "scripts/build_combined_kaggle_package.py",
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def _code_source(notebook: dict) -> str:
    return "\n".join(
        BUILDER._cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class CombinedKagglePackageTests(unittest.TestCase):
    def test_active_notebook_has_one_supported_magic_and_compiles(self) -> None:
        notebook = BUILDER._load_candidate_notebook()
        self.assertEqual(BUILDER._magic_cell_indices(notebook), [2])
        compiled = BUILDER._compile_notebook(notebook, label="test")
        self.assertGreater(len(compiled), 1)
        self.assertNotIn(2, compiled)

    def test_preflight_injection_is_at_the_offline_game_anchor(self) -> None:
        notebook = BUILDER._load_candidate_notebook()
        BUILDER._inject_preflight_limit(notebook, 3)
        source = _code_source(notebook)
        self.assertIn("bm.games = bm.games[:3]", source)
        self.assertEqual(source.count("bm.games = bm.games[:3]"), 1)
        BUILDER._compile_notebook(notebook, label="preflight")

    def test_full_notebook_is_not_injected(self) -> None:
        notebook = BUILDER._load_candidate_notebook()
        source = _code_source(notebook)
        self.assertNotIn("bm.games = bm.games[:", source)

    def test_unowned_output_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as tmp:
            output_dir = Path(tmp) / "unowned"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("user-owned\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unowned output directory"):
                BUILDER._owned_output_dir(output_dir)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user-owned\n")

    def test_metadata_matches_active_inputs(self) -> None:
        metadata = BUILDER._metadata(
            kernel_id="izukia/arc-agi3-m2-combined-profile-preflight",
            title="arc agi3 m2 combined profile preflight",
            notebook_name="arc-agi3-m2-combined-profile-preflight.ipynb",
        )
        BUILDER._validate_metadata(
            metadata,
            notebook_name="arc-agi3-m2-combined-profile-preflight.ipynb",
        )
        self.assertEqual(
            metadata["dataset_sources"],
            ["dfranzen/taaf-kaggle-source-bundle-copy", "dfranzen/pennyroyal-v253"],
        )
        self.assertEqual(
            metadata["model_sources"],
            [
                "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
                "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
            ],
        )

    def test_build_package_writes_manifest_and_reproducible_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as tmp:
            output_dir = Path(tmp) / "candidate"
            args = SimpleNamespace(
                kernel_id="izukia/arc-agi3-m2-combined-profile-preflight",
                title=None,
                mode="preflight",
                output_dir=output_dir,
                offline_max_games=1,
            )
            built = BUILDER.build_package(args)
            self.assertEqual(built, output_dir.resolve())
            notebook_path = built / "arc-agi3-m2-combined-profile-preflight.ipynb"
            metadata_path = built / "kernel-metadata.json"
            manifest_path = built / "build-manifest.json"
            self.assertTrue((built / BUILDER.PACKAGE_MARKER).is_file())
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["code_file"], notebook_path.name)
            self.assertEqual(manifest["mode"], "preflight")
            self.assertEqual(manifest["offline_max_games"], 1)
            self.assertIn("bm.games = bm.games[:1]", _code_source(notebook))
            BUILDER._compile_notebook(notebook, label="generated-combined-code")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
