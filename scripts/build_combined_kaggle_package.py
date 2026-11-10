#!/usr/bin/env python3
"""Build a validated Kaggle package for the active combined ARC-AGI-3 notebook.

This script only prepares local package files. It never authenticates with
Kaggle, uploads a kernel, starts a run, or creates a competition submission.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "arc-agi-3-milestone-2-solution-combined.ipynb"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"

COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"
DATASET_SOURCES = [
    "dfranzen/taaf-kaggle-source-bundle-copy",
    "dfranzen/pennyroyal-v253",
]
MODEL_SOURCES = [
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
    "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
]
MACHINE_SHAPE = "NvidiaRtxPro6000"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)
PACKAGE_MARKER = ".arc-agi-kaggle-package"
EXPECTED_MAGIC_PREFIX = "%%writefile /kaggle/harness-changes.patch"
OFFLINE_GAME_ANCHOR = "    bm.games = _offline_games(competition_env_files)\n"
EXPECTED_NOTEBOOK_ANCHORS = (
    "ARC3_CROSS_SOLUTION_PROFILE",
    "_find_bundle_dir",
    "WHEELHOUSE_DIR",
    "MODEL_DIR",
    "DRAFT_MODEL_DIR",
    'TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN"',
    OFFLINE_GAME_ANCHOR.rstrip("\n"),
    "await bm.run(soft_end_time=soft_end, runtime_environment=target",
    "Offline validation is diagnostic only",
    "_validate_benchmark_output",
    "expected_columns = [\"row_id\", \"game_id\", \"end_of_game\", \"score\"]",
    "parquet_file.read(columns=expected_columns).to_pylist()",
    "duplicate row_id values",
    "require_rows=TRUE_SUBMISSION",
    "combined_run_contract.json",
    "bm.solver.max_runtime_s_per_game = 532*60",
    "bm.solver.max_runtime_s_per_game = 25*60",
)


def _slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def _parse_kernel_id(value: str) -> tuple[str, str]:
    parts = value.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "--kernel-id must have the form <kaggle-username>/<kernel-slug>."
        )
    owner, slug = parts
    if _slugify(owner) != owner or _slugify(slug) != slug:
        raise ValueError(
            "Kaggle username and kernel slug must already be lowercase URL slugs."
        )
    return owner, slug


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _magic_cell_indices(notebook: dict) -> list[int]:
    indices = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell).lstrip()
        if source.startswith("%%"):
            indices.append(index)
    return indices


def _compile_notebook(notebook: dict, *, label: str) -> list[int]:
    """Compile ordinary code cells, explicitly allowing the patch magic cell."""

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError(f"{label} has no notebook cell list.")
    compiled = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        stripped = source.lstrip()
        if stripped.startswith("%%"):
            if not stripped.startswith(EXPECTED_MAGIC_PREFIX):
                raise SyntaxError(
                    f"{label}:cell-{index} contains an unsupported cell magic."
                )
            continue
        if stripped.startswith(("%", "!")):
            raise SyntaxError(
                f"{label}:cell-{index} contains an unsupported IPython command."
            )
        compile(
            source,
            f"{label}:cell-{index}",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
        compiled.append(index)
    if not compiled:
        raise ValueError(f"{label} has no ordinary Python code cells.")
    return compiled


def _clear_execution_artifacts(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []


def _inject_preflight_limit(notebook: dict, max_games: int) -> None:
    """Limit only the offline package's game list at the active runtime anchor."""

    if max_games < 1:
        raise ValueError("max_games must be at least 1")
    injection = (
        "\n"
        "    # Package-only Save & Run preflight limit; competition reruns use the gateway.\n"
        f"    bm.games = bm.games[:{max_games}]\n"
    )
    matches = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        if OFFLINE_GAME_ANCHOR not in source:
            continue
        matches += source.count(OFFLINE_GAME_ANCHOR)
        source = source.replace(OFFLINE_GAME_ANCHOR, OFFLINE_GAME_ANCHOR + injection)
        cell["source"] = source.splitlines(keepends=True)
    if matches != 1:
        raise ValueError(
            f"Preflight injection anchor matched {matches} times; expected once."
        )


def _load_candidate_notebook() -> dict:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Active combined notebook must use nbformat 4.")
    source_text = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))
    missing = [anchor for anchor in EXPECTED_NOTEBOOK_ANCHORS if anchor not in source_text]
    if missing:
        raise ValueError(
            "Active combined notebook is missing required anchors: "
            + ", ".join(missing)
        )
    magic_cells = _magic_cell_indices(notebook)
    if len(magic_cells) != 1:
        raise ValueError(
            f"Expected exactly one patch magic cell; found indices {magic_cells}."
        )
    _compile_notebook(notebook, label=str(SOURCE_NOTEBOOK))
    return notebook


def _metadata(*, kernel_id: str, title: str, notebook_name: str) -> dict:
    _, slug = _parse_kernel_id(kernel_id)
    if _slugify(title) != slug:
        raise ValueError(
            f"Kaggle links titles to slugs: title slug {_slugify(title)!r} must equal {slug!r}."
        )
    return {
        "id": kernel_id,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION_SOURCE],
        "kernel_sources": [],
        "model_sources": MODEL_SOURCES,
        "docker_image": DOCKER_IMAGE,
        "machine_shape": MACHINE_SHAPE,
    }


def _validate_metadata(metadata: dict, *, notebook_name: str) -> None:
    expected = {
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION_SOURCE],
        "kernel_sources": [],
        "model_sources": MODEL_SOURCES,
        "docker_image": DOCKER_IMAGE,
        "machine_shape": MACHINE_SHAPE,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Generated Kaggle metadata mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owned_output_dir(output_dir: Path) -> None:
    if output_dir == REPO_ROOT or REPO_ROOT not in output_dir.parents:
        raise ValueError("--output-dir must stay inside this repository.")
    if not output_dir.exists():
        return
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError(f"Refusing to replace non-directory output path: {output_dir}")
    marker = output_dir / PACKAGE_MARKER
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != "owned"
    ):
        raise ValueError(
            f"Refusing to replace unowned output directory: {output_dir}. "
            f"Expected marker {PACKAGE_MARKER!r}."
        )
    shutil.rmtree(output_dir)


def build_package(args: argparse.Namespace) -> Path:
    _, slug = _parse_kernel_id(args.kernel_id)
    title = args.title or slug.replace("-", " ")
    output_dir = (
        args.output_dir
        or DEFAULT_BUILD_ROOT / f"combined-kaggle-{args.mode}"
    ).resolve()

    notebook = _load_candidate_notebook()
    if args.mode == "preflight":
        _inject_preflight_limit(notebook, args.offline_max_games)
    _clear_execution_artifacts(notebook)
    compiled_cells = _compile_notebook(notebook, label=f"generated-{args.mode}")
    magic_cells = _magic_cell_indices(notebook)

    notebook_name = f"arc-agi3-m2-combined-profile-{args.mode}.ipynb"
    metadata = _metadata(
        kernel_id=args.kernel_id, title=title, notebook_name=notebook_name
    )
    _validate_metadata(metadata, notebook_name=notebook_name)

    _owned_output_dir(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / PACKAGE_MARKER).write_text("owned\n", encoding="utf-8")
    notebook_path = output_dir / notebook_name
    metadata_path = output_dir / "kernel-metadata.json"
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Re-read exactly what will be handed to Kaggle.
    written_notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    written_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    written_compiled = _compile_notebook(written_notebook, label=str(notebook_path))
    _validate_metadata(written_metadata, notebook_name=notebook_name)
    if written_compiled != compiled_cells:
        raise ValueError(
            f"Compiled-cell set changed after writing: {compiled_cells} -> {written_compiled}"
        )

    digest = _sha256(notebook_path)
    manifest = {
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(SOURCE_NOTEBOOK),
        "generated_notebook": notebook_name,
        "generated_sha256": digest,
        "mode": args.mode,
        "offline_max_games": args.offline_max_games if args.mode == "preflight" else None,
        "compiled_code_cells": compiled_cells,
        "skipped_magic_cells": magic_cells,
        "metadata": written_metadata,
    }
    (output_dir / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Built active combined Kaggle {args.mode} package: {output_dir}")
    print(f"Notebook SHA-256: {digest}")
    print(f"Source SHA-256: {manifest['source_sha256']}")
    print(f"Compiled code cells: {compiled_cells}; skipped magic cells: {magic_cells}")
    print(f"Machine shape: {MACHINE_SHAPE}; internet: disabled")
    print("No Kaggle authentication, upload, run, or submission was performed.")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-id", required=True, help="Private target as <username>/<slug>."
    )
    parser.add_argument("--mode", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--title", help="Defaults to the kernel slug with spaces.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Defaults to build/combined-kaggle-<mode>.",
    )
    parser.add_argument(
        "--offline-max-games",
        type=int,
        default=1,
        help="Offline game limit injected only into preflight packages (default: 1).",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.offline_max_games < 1:
        parser.error("--offline-max-games must be at least 1.")
    build_package(args)


if __name__ == "__main__":
    main()
