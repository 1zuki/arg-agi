#!/usr/bin/env python3
"""Build a validated Kaggle kernel package without uploading or running it."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "arc-agi.ipynb"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"

COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"
DATASET_SOURCES = [
    "jeroencottaar/taaf-kaggle-source-share",
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot",
]
MACHINE_SHAPE = "NvidiaRtxPro6000"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)
EXPECTED_NOTEBOOK_ANCHORS = (
    'SUBMISSION_VARIANT = "taaf-qwen38-xhigh-checkpoint8-undo-sync"',
    'MODEL_HF_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"',
    'MODEL_DATASET_REF = "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"',
    "EXPECTED_SOURCE_TREE_SHA256",
    "EXPECTED_WHEELHOUSE_METADATA_SHA256",
    "EXPECTED_MODEL_FILE_COUNT = 80",
    'ACTION7_MODEL_LABEL = "UNDO"',
    "AUTO_RESET_CONTEXT_SYNC = True",
    "_write_submission_health_summary()",
)
PACKAGE_MARKER = ".arc-agi-kaggle-package"


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


def _compile_notebook(notebook: dict, *, label: str) -> list[int]:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError(f"{label} has no notebook cell list.")
    compiled = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(
            source,
            f"{label}:cell-{index}",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
        compiled.append(index)
    if not compiled:
        raise ValueError(f"{label} has no code cells.")
    return compiled


def _clear_execution_artifacts(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []


def _inject_preflight_limit(notebook: dict, max_games: int) -> None:
    anchor = "NOTEBOOK_START_EPOCH = time.time()\n"
    injection = (
        "\n"
        "# Package-only Save & Run preflight limit. Competition reruns ignore this knob.\n"
        "if not TRUE_SUBMISSION:\n"
        f'    os.environ.setdefault("TAAF_OFFLINE_MAX_GAMES", "{max_games}")\n'
    )
    matches = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if anchor not in source:
            continue
        matches += source.count(anchor)
        source = source.replace(anchor, anchor + injection)
        cell["source"] = source.splitlines(keepends=True)
    if matches != 1:
        raise ValueError(
            f"Preflight injection anchor matched {matches} times; expected once."
        )


def _load_candidate_notebook() -> dict:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
    )
    missing = [
        anchor for anchor in EXPECTED_NOTEBOOK_ANCHORS if anchor not in source_text
    ]
    if missing:
        raise ValueError(
            "Candidate notebook is missing required anchors: " + ", ".join(missing)
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
        "model_sources": [],
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
        "enable_internet": False,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION_SOURCE],
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


def build_package(args: argparse.Namespace) -> Path:
    _, slug = _parse_kernel_id(args.kernel_id)
    title = args.title or slug.replace("-", " ")
    output_dir = (
        args.output_dir or DEFAULT_BUILD_ROOT / f"kaggle-{args.mode}"
    ).resolve()
    if output_dir == REPO_ROOT or REPO_ROOT not in output_dir.parents:
        raise ValueError("--output-dir must stay inside this repository.")

    notebook = _load_candidate_notebook()
    if args.mode == "preflight":
        _inject_preflight_limit(notebook, args.offline_max_games)
    _clear_execution_artifacts(notebook)
    compiled_cells = _compile_notebook(notebook, label=f"generated-{args.mode}")

    notebook_name = f"arc-agi3-qwen38-checkpoint8-undo-sync-{args.mode}.ipynb"
    metadata = _metadata(
        kernel_id=args.kernel_id, title=title, notebook_name=notebook_name
    )
    _validate_metadata(metadata, notebook_name=notebook_name)

    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(
                f"Refusing to replace non-directory output path: {output_dir}"
            )
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
    _compile_notebook(written_notebook, label=str(notebook_path))
    _validate_metadata(written_metadata, notebook_name=notebook_name)

    digest = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    print(f"Built private Kaggle {args.mode} package: {output_dir}")
    print(f"Notebook SHA-256: {digest}")
    print(f"Compiled code cells: {compiled_cells}")
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
        "--output-dir", type=Path, help="Defaults to build/kaggle-<mode>."
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
