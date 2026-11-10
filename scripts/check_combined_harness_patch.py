#!/usr/bin/env python3
"""Verify the combined notebook's embedded harness patch against a clean base.

The checker reads a notebook cell and a commit from an external Git checkout,
then performs all work in a temporary tree. It never edits the external
checkout and never applies the patch to a persistent working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "arc-agi-3-milestone-2-solution-combined.ipynb"
DEFAULT_EXTERNAL_REPO = REPO_ROOT / "external" / "arc-agi-3-solution"
DEFAULT_BASE_COMMIT = "7652836056c59e044f093e3c13ed7438c814169e"
DEFAULT_PATCH_CELL = 2
PATCH_MAGIC = "%%writefile /kaggle/harness-changes.patch"
LEGACY_SEED_LINE = "            seed=_LOCAL_ANALYZER_SEED,\n"


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def extract_patch(notebook_path: Path, *, cell_index: int = DEFAULT_PATCH_CELL) -> bytes:
    """Return the exact patch payload after the notebook write-file magic."""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not 0 <= cell_index < len(cells):
        raise ValueError(f"Notebook has no cell at index {cell_index}.")
    source = _cell_source(cells[cell_index])
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != PATCH_MAGIC:
        raise ValueError(
            f"Notebook cell {cell_index} must begin with {PATCH_MAGIC!r}."
        )
    patch = "".join(lines[1:]).encode("utf-8")
    if not patch.startswith(b"diff --git "):
        raise ValueError("Notebook patch payload does not start with a unified diff.")
    return patch


def normalize_legacy_seed(base_tree: Path) -> int:
    """Apply the notebook's one-line compatibility normalization to a temp tree."""

    tool_agent = (
        base_tree
        / "ARC3-Inference"
        / "inference"
        / "agent"
        / "tool_agent.py"
    )
    if not tool_agent.is_file():
        raise FileNotFoundError(tool_agent)
    source = tool_agent.read_text(encoding="utf-8")
    count = source.count(LEGACY_SEED_LINE)
    if count > 1:
        raise ValueError(
            "Temporary baseline contains duplicate legacy request-seed lines."
        )
    if count == 1:
        tool_agent.write_text(source.replace(LEGACY_SEED_LINE, "", 1), encoding="utf-8")
    return count


def _extract_archive(repo: Path, commit: str, destination: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit, "ARC3-Inference"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git archive failed for {commit}: {detail}")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        destination_resolved = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member path: {member.name!r}")
            archive.extract(member, destination)


def check_patch(
    *,
    notebook_path: Path,
    external_repo: Path,
    base_commit: str,
    patch_cell: int = DEFAULT_PATCH_CELL,
) -> dict:
    """Return a machine-readable patch provenance result."""

    patch = extract_patch(notebook_path, cell_index=patch_cell)
    with tempfile.TemporaryDirectory(prefix="arc3-harness-patch-") as temp_dir:
        base_tree = Path(temp_dir)
        _extract_archive(external_repo, base_commit, base_tree)
        normalized_count = normalize_legacy_seed(base_tree)
        patch_path = base_tree / "harness-changes.patch"
        patch_path.write_bytes(patch)
        applied = subprocess.run(
            ["git", "apply", "--check", "--no-index", str(patch_path)],
            cwd=base_tree,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return {
        "schema_version": 1,
        "base_commit": base_commit,
        "base_source": str(external_repo),
        "notebook": str(notebook_path),
        "patch_cell_index": patch_cell,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "legacy_seed_normalization_count": normalized_count,
        "git_apply_check": applied.returncode == 0,
        "stdout": applied.stdout.decode("utf-8", errors="replace").strip(),
        "stderr": applied.stderr.decode("utf-8", errors="replace").strip(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--external-repo", type=Path, default=DEFAULT_EXTERNAL_REPO)
    parser.add_argument("--base-commit", default=DEFAULT_BASE_COMMIT)
    parser.add_argument("--patch-cell", type=int, default=DEFAULT_PATCH_CELL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    notebook = args.notebook.resolve()
    external_repo = args.external_repo.resolve()
    output = args.output.resolve()
    if not notebook.is_file():
        raise SystemExit(f"Notebook not found: {notebook}")
    if not (external_repo / ".git").exists():
        raise SystemExit(f"External Git checkout not found: {external_repo}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing report: {output}")
    report = check_patch(
        notebook_path=notebook,
        external_repo=external_repo,
        base_commit=args.base_commit,
        patch_cell=args.patch_cell,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["git_apply_check"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
