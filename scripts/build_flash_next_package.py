#!/usr/bin/env python3
"""Build the isolated public Flash-Next/MTP ARC-AGI-3 Kaggle packages."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "arc-agi-flash-next-mtp.ipynb"
BUILD_ROOT = REPO_ROOT / "build"
PACKAGE_MARKER = ".arc-agi-kaggle-package"

COMPETITION_SOURCE = "arc-prize-2026-arc-agi-3"
DATASET_SOURCES = [
    "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
    "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
]
MODEL_SOURCES = [
    "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1",
]
MACHINE_SHAPE = "NvidiaRtxPro6000"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)
VARIANT = "taaf-qwen38-flash-next-mtp"


def _slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def _parse_kernel_id(value: str) -> tuple[str, str]:
    parts = value.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--kernel-id must have the form <username>/<slug>.")
    owner, slug = parts
    if _slugify(owner) != owner or _slugify(slug) != slug:
        raise ValueError("Kaggle username and slug must be lowercase URL slugs.")
    return owner, slug


def _compile_notebook(notebook: dict, label: str) -> list[int]:
    compiled: list[int] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{label}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        compiled.append(index)
    if not compiled:
        raise ValueError(f"{label} has no code cells.")
    return compiled


def _clean_execution(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Flash notebook anchor {label!r} matched {count} times; expected once.")
    return text.replace(old, new)


def _add_preflight_support(notebook: dict, *, max_games: int | None) -> None:
    start_anchor = "NOTEBOOK_START_EPOCH = time.time()\n"
    start_injection = (
        "\n"
        "FLASH_OFFLINE_MAX_GAMES = int(os.environ.get(\"FLASH_OFFLINE_MAX_GAMES\", \"0\"))\n"
        "FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get(\"FLASH_OFFLINE_MAX_RUNTIME_S\", \"0\"))\n"
    )
    matches = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if start_anchor in source:
            matches += source.count(start_anchor)
            source = source.replace(start_anchor, start_anchor + start_injection)
            cell["source"] = source.splitlines(keepends=True)
    if matches != 1:
        raise ValueError(f"Flash notebook start anchor matched {matches} times; expected once.")

    list_anchor = "])\n\nif TRUE_SUBMISSION:\n"
    list_injection = (
        "])\n\n"
        "if not TRUE_SUBMISSION and FLASH_OFFLINE_MAX_GAMES:\n"
        "    PUBLIC_GAME_IDS = PUBLIC_GAME_IDS[:FLASH_OFFLINE_MAX_GAMES]\n\n"
        "if TRUE_SUBMISSION:\n"
    )
    list_matches = 0
    guard_matches = 0
    coverage_matches = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if list_anchor in source:
            list_matches += source.count(list_anchor)
            source = source.replace(list_anchor, list_injection)
        guard = "if len(bm.games) != 25:\n"
        if guard in source:
            guard_matches += source.count(guard)
            source = source.replace(guard, "if len(bm.games) != len(PUBLIC_GAME_IDS):\n")
        coverage = "if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):"
        if coverage in source:
            coverage_matches += source.count(coverage)
            source = source.replace(
                coverage,
                "if len(public_runs) != len(PUBLIC_GAME_IDS) or public_run_ids != list(PUBLIC_GAME_IDS):",
            )
        extra_check = "    extra = sorted(set(offline_by_id) - set(PUBLIC_GAME_IDS))\n"
        if extra_check in source:
            source = source.replace(
                extra_check,
                "    extra = [] if (not TRUE_SUBMISSION and FLASH_OFFLINE_MAX_GAMES) else sorted(set(offline_by_id) - set(PUBLIC_GAME_IDS))\n",
                1,
            )
        cell["source"] = source.splitlines(keepends=True)
    if list_matches != 1 or guard_matches != 1 or coverage_matches != 1:
        raise ValueError(
            "Flash notebook public-game anchors did not match exactly once: "
            f"list={list_matches}, guard={guard_matches}, coverage={coverage_matches}."
        )

    teardown_old = """            subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=WORKING_DIR,
                env=_command_env(),
                timeout=30.0,
            )"""
    teardown_new = """            subprocess.run(
                [sys.executable, str(FLASH_TEARDOWN_PATH)],
                check=True,
                cwd=WORKING_DIR,
                env=_command_env(),
                timeout=45.0,
            )"""
    teardown_matches = 0
    for candidate in notebook["cells"]:
        if candidate.get("cell_type") != "code":
            continue
        text = "".join(candidate.get("source", []))
        teardown_matches += text.count(teardown_old)
        text = text.replace(teardown_old, teardown_new)
        text = text.replace("PUBLIC25_AUDIT runs=25 actions=", "PUBLIC25_AUDIT runs={len(public_runs)} actions=")
        candidate["source"] = text.splitlines(keepends=True)
    if teardown_matches != 1:
        raise ValueError(f"Flash teardown anchor matched {teardown_matches} times; expected once.")

    # Stage the verified patch before GPU setup, not after an expensive run.
    setup_anchor = "# Each bundled repo exposes its importable tree at <repo>/src or <repo>.\n"
    patch_source = (REPO_ROOT / "scripts/flash_teardown_patch.py").read_text(encoding="utf-8")
    stage_source = patch_source + """
import hashlib

if json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()) != [
    '\"$PYTHON\" \"$TAAF_KAGGLE_BUNDLE_DIR/serving_teardown.py\"'
]:
    raise RuntimeError("Flash teardown command contract changed.")
_teardown_original = (BUNDLE_DIR / "serving_teardown.py").read_bytes()
_teardown_patched = patch_flash_teardown(_teardown_original)
FLASH_TEARDOWN_PATH = WORKING_DIR / "flash_serving_teardown.py"
FLASH_TEARDOWN_PATH.write_text(_teardown_patched, encoding="utf-8")
_teardown_patch_record = {
    "patch": "gpu-release-settle-v1",
    "source_sha256": hashlib.sha256(_teardown_original).hexdigest(),
    "patched_sha256": hashlib.sha256(_teardown_patched.encode("utf-8")).hexdigest(),
}
(WORKING_DIR / "flash_teardown_patch.json").write_text(
    json.dumps(_teardown_patch_record, indent=2) + "\\n", encoding="utf-8"
)
print("FLASH_TEARDOWN_PATCH", json.dumps(_teardown_patch_record), flush=True)

"""
    setup_matches = 0
    for candidate in notebook["cells"]:
        if candidate.get("cell_type") != "code":
            continue
        text = "".join(candidate.get("source", []))
        setup_matches += text.count(setup_anchor)
        text = text.replace(setup_anchor, stage_source + setup_anchor)
        candidate["source"] = text.splitlines(keepends=True)
    if setup_matches != 1:
        raise ValueError(f"Flash setup anchor matched {setup_matches} times; expected once.")

    if max_games is not None:
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "FLASH_OFFLINE_MAX_GAMES" in "".join(cell.get("source", []))
        )
        source = "".join(cell["source"])
        source = source.replace(
            'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "0"))',
            f'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "{max_games}"))',
            1,
        )
        source = source.replace(
            'FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "0"))',
            'FLASH_OFFLINE_MAX_RUNTIME_S = 0.0 if TRUE_SUBMISSION else float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "1800"))',
            1,
        )
        cell["source"] = source.splitlines(keepends=True)

        settings_anchor = "bm.solver.max_runtime_s_per_game = 7920.0"
        budget_anchor = 'budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)'
        replaced_settings = 0
        replaced_budget = 0
        for candidate in notebook["cells"]:
            if candidate.get("cell_type") != "code":
                continue
            candidate_source = "".join(candidate.get("source", []))
            if settings_anchor in candidate_source:
                replaced_settings += candidate_source.count(settings_anchor)
                candidate_source = candidate_source.replace(
                    settings_anchor,
                    "bm.solver.max_runtime_s_per_game = (FLASH_OFFLINE_MAX_RUNTIME_S "
                    "if FLASH_OFFLINE_MAX_RUNTIME_S > 0 else 7920.0)",
                )
            if budget_anchor in candidate_source:
                replaced_budget += candidate_source.count(budget_anchor)
                candidate_source = candidate_source.replace(
                    budget_anchor,
                    'budget = min(float(getattr(target, "max_runtime_s", 0.0) or 0.0), '
                    'FLASH_OFFLINE_MAX_RUNTIME_S + 600.0) if FLASH_OFFLINE_MAX_RUNTIME_S > 0 '
                    'else float(getattr(target, "max_runtime_s", 0.0) or 0.0)',
                )
            candidate["source"] = candidate_source.splitlines(keepends=True)
        if replaced_settings != 1 or replaced_budget != 1:
            raise ValueError(
                "Flash preflight runtime anchors did not match exactly once: "
                f"settings={replaced_settings}, budget={replaced_budget}."
            )


def _metadata(kernel_id: str, notebook_name: str, title: str) -> dict:
    _parse_kernel_id(kernel_id)
    if _slugify(title) != _parse_kernel_id(kernel_id)[1]:
        raise ValueError("Kaggle metadata title must slugify to the kernel slug.")
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


def build(args: argparse.Namespace) -> Path:
    _parse_kernel_id(args.kernel_id)
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    _add_preflight_support(notebook, max_games=1 if args.mode == "preflight" else None)
    _clean_execution(notebook)
    notebook_name = f"arc-agi3-qwen38-flash-next-mtp-{args.mode}.ipynb"
    title = args.title or _parse_kernel_id(args.kernel_id)[1].replace("-", " ")
    metadata = _metadata(args.kernel_id, notebook_name, title)
    compiled = _compile_notebook(notebook, f"generated-{args.mode}")

    output = (args.output_dir or BUILD_ROOT / f"kaggle-flash-next-{args.mode}").resolve()
    if output == REPO_ROOT or REPO_ROOT not in output.parents:
        raise ValueError("--output-dir must stay inside this repository.")
    if output.exists():
        marker = output / PACKAGE_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "owned":
            raise ValueError(f"Refusing to replace unowned output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / PACKAGE_MARKER).write_text("owned\n", encoding="utf-8")
    notebook_path = output / notebook_name
    metadata_path = output / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _compile_notebook(json.loads(notebook_path.read_text(encoding="utf-8")), str(notebook_path))
    print(f"Built Flash-Next/MTP Kaggle {args.mode} package: {output}")
    print(f"Compiled code cells: {compiled}")
    print("No Kaggle upload, run, or submission was performed.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
