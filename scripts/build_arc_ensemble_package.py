#!/usr/bin/env python3
"""Build the isolated Flash-primary ensemble controller package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from arc_ensemble_patch import (
    HYBRID_REFERENCE_SHA256,
    PULLED_HYBRID_REFERENCE_SHA256,
    PATCH_NAME,
    SOLVER_SOURCE_SHA256,
    inspect_hybrid_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "arc-agi3-flash-next-mtp-full.ipynb"
ACTIVE_SOLVER = REPO_ROOT / "build/flash-baseline-source-20260920/solver.py"
HYBRID_REFERENCE = Path.home() / "Downloads/arc_agi_3_hybrid_repl_agent.py"
PACKAGE_MARKER = ".arc-agi-ensemble-package"
COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"
DATASET_SOURCES = [
    "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
    "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
]
MODEL_SOURCES = [
    "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1",
]
SOURCE_NOTEBOOK_SHA256 = (
    "97b56617890ccb534a1e35aa127c8af6b50ca31964eb6e0241c8b85c2ca966cc"
)


def _compile_notebook(notebook: dict, label: str) -> list[int]:
    compiled: list[int] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{label}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        compiled.append(index)
    if not compiled:
        raise ValueError(f"{label} has no code cells")
    return compiled


def _clean_execution(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"ensemble notebook anchor {label!r} matched {count} times")
    return text.replace(old, new, 1)


def _inject(notebook: dict, *, controller_source: str, patch_source: str, hybrid_receipt: dict) -> None:
    intro = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown")
    intro["source"] = (
        "## Experimental Flash-primary / Hybrid-recovery controller\n\n"
        "Not a verified score improvement. Protected incumbent: Flash-Next/MTP c28, 3.40, "
        "submission 56170646. Flash settings and model-serving profile are unchanged. "
        "The pulled Hybrid reference is launcher-only; recovery fails closed until a genuine "
        "adapter is supplied.\n\n"
        + "".join(intro.get("source", []))
    ).splitlines(keepends=True)

    anchor = 'print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")\n'
    stage = f"""

# Digest-bound ensemble overlay. The original input bundle remains read-only.
import hashlib
import json
import sys

ENSEMBLE_PATCH_NAME = {PATCH_NAME!r}
ENSEMBLE_HYBRID_RECEIPT = {hybrid_receipt!r}
ENSEMBLE_OVERLAY_ROOT = WORKING_DIR / "flash_ensemble_overlay"
ENSEMBLE_OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)
(ENSEMBLE_OVERLAY_ROOT / "arc_ensemble_controller.py").write_text(
    {controller_source!r}, encoding="utf-8"
)
_patch_namespace = {{}}
exec(compile({patch_source!r}, "arc_ensemble_patch.py", "exec"), _patch_namespace)
_patch_solver = _patch_namespace["patch_solver"]
_solver_source_path = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "framework" / "solver.py"
if _solver_source_path.is_symlink() or not _solver_source_path.is_file():
    raise RuntimeError("Flash solver source is missing or a symlink.")
_solver_original = _solver_source_path.read_bytes()
_solver_patched = _patch_solver(_solver_original)
if hashlib.sha256(_solver_original).hexdigest() != {SOLVER_SOURCE_SHA256!r}:
    raise RuntimeError("Active Flash solver digest does not match the pinned source anchor.")

_inference_root = ENSEMBLE_OVERLAY_ROOT / "inference"
_framework_root = _inference_root / "framework"
_framework_root.mkdir(parents=True, exist_ok=True)
(_inference_root / "__init__.py").write_text(
    "from pkgutil import extend_path\\n__path__ = extend_path(__path__, __name__)\\n",
    encoding="utf-8",
)
_framework_init = _solver_source_path.parent / "__init__.py"
_framework_text = _framework_init.read_text(encoding="utf-8") if _framework_init.is_file() else ""
(_framework_root / "__init__.py").write_text(
    "from pkgutil import extend_path\\n__path__ = extend_path(__path__, __name__)\\n" + _framework_text,
    encoding="utf-8",
)
_solver_overlay_path = _framework_root / "solver.py"
_solver_overlay_path.write_text(_solver_patched, encoding="utf-8")
sys.path.insert(0, str(ENSEMBLE_OVERLAY_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [str(ENSEMBLE_OVERLAY_ROOT), os.environ.get("PYTHONPATH", "")] if entry
)
_solver_patch_record = {{
    "patch": ENSEMBLE_PATCH_NAME,
    "source_sha256": hashlib.sha256(_solver_original).hexdigest(),
    "patched_sha256": hashlib.sha256(_solver_patched.encode("utf-8")).hexdigest(),
    "hybrid_reference": ENSEMBLE_HYBRID_RECEIPT,
    "adapter_available": False,
}}
(WORKING_DIR / "flash_ensemble_patch.json").write_text(
    json.dumps(_solver_patch_record, indent=2) + "\\n", encoding="utf-8"
)
print("FLASH_ENSEMBLE_OVERLAY_READY", json.dumps(_solver_patch_record), flush=True)
"""
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if anchor in source:
            cell["source"] = _replace_once(source, anchor, anchor + stage, "overlay staging").splitlines(keepends=True)
            break
    else:
        raise ValueError("ensemble source-path anchor was not found")


def build(output: Path) -> Path:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    if not ACTIVE_SOLVER.is_file():
        raise FileNotFoundError(ACTIVE_SOLVER)
    if not HYBRID_REFERENCE.is_file():
        raise FileNotFoundError(HYBRID_REFERENCE)
    if hashlib.sha256(ACTIVE_SOLVER.read_bytes()).hexdigest() != SOLVER_SOURCE_SHA256:
        raise ValueError("local active solver does not match the pinned digest")
    if hashlib.sha256(SOURCE_NOTEBOOK.read_bytes()).hexdigest() != SOURCE_NOTEBOOK_SHA256:
        raise ValueError("active Flash notebook does not match the pinned digest")
    hybrid_receipt = inspect_hybrid_reference(HYBRID_REFERENCE.read_bytes())
    if hybrid_receipt["sha256"] not in {
        HYBRID_REFERENCE_SHA256,
        PULLED_HYBRID_REFERENCE_SHA256,
    }:
        raise ValueError("Hybrid reference receipt changed")

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    _inject(
        notebook,
        controller_source=(REPO_ROOT / "scripts/arc_ensemble_controller.py").read_text(encoding="utf-8"),
        patch_source=(REPO_ROOT / "scripts/arc_ensemble_patch.py").read_text(encoding="utf-8"),
        hybrid_receipt=hybrid_receipt,
    )
    _clean_execution(notebook)
    compiled = _compile_notebook(notebook, "arc-ensemble")

    output = output.resolve()
    if output == REPO_ROOT or REPO_ROOT not in output.parents:
        raise ValueError("output must stay inside this repository")
    if output.exists():
        marker = output / PACKAGE_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "owned":
            raise ValueError(f"refusing to replace unowned output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / PACKAGE_MARKER).write_text("owned\n", encoding="utf-8")
    notebook_path = output / "arc-agi3-flash-next-mtp-ensemble-controller.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "hybrid-reference-receipt.json").write_text(
        json.dumps(hybrid_receipt, indent=2) + "\n", encoding="utf-8"
    )
    (output / "ensemble-source-receipt.json").write_text(
        json.dumps(
            {
                "patch": PATCH_NAME,
                "active_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
                "active_solver_sha256": SOLVER_SOURCE_SHA256,
                "hybrid_reference_sha256": hybrid_receipt["sha256"],
                "hybrid_adapter_available": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "your-kaggle-username/arc-agi3-flash-next-mtp-ensemble-controller",
                "title": "arc agi3 flash next mtp ensemble controller",
                "code_file": notebook_path.name,
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
                "adapter_available": False,
                "protected_incumbent": {"submission": "56170646", "score": 3.40},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _compile_notebook(json.loads(notebook_path.read_text(encoding="utf-8")), str(notebook_path))
    print(f"Built isolated ensemble controller package: {output}")
    print(f"Compiled code cells: {compiled}")
    print("No Kaggle upload, run, or competition submission was performed.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build/arc-ensemble-controller-preflight",
    )
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
