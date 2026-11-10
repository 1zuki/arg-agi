#!/usr/bin/env python3
"""Build a c28 control or zero-change-repeat-guard candidate from the pulled 3.40 notebook."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from build_flash_next_package import (
    BUILD_ROOT,
    PACKAGE_MARKER,
    REPO_ROOT,
    _metadata,
    _parse_kernel_id,
)


SOURCE_NOTEBOOK = REPO_ROOT / "arc-agi3-flash-next-mtp-full.ipynb"
PATCH_SOURCE = REPO_ROOT / "scripts/flash_zero_change_repeat_guard_patch.py"


def _compile_notebook(notebook: dict, label: str) -> list[int]:
    compiled = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
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


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Pulled Flash anchor {label!r} matched {count} times.")
    return source.replace(old, new, 1)


def _cell(notebook: dict, anchor: str) -> tuple[dict, str]:
    matches = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if anchor in source:
            matches.append((cell, source))
    if len(matches) != 1:
        raise ValueError(f"Pulled Flash cell anchor {anchor!r} matched {len(matches)} times.")
    return matches[0]


def patch_notebook(notebook: dict, *, control: bool) -> None:
    intro = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown")
    if control:
        heading = (
            "## Exact c28 control replay from the protected 3.40 notebook\n\n"
            "This package intentionally preserves the downloaded/audited 3.40 c28 behavior. "
            "It is a private stability control, not a score improvement.\n\n"
        )
    else:
        heading = (
            "## Experimental zero-change repeat guard candidate\n\n"
            "Based on the downloaded 3.40 Flash notebook. The only solver-policy change is "
            "a digest-bound guard that blocks one singleton action repeated immediately after "
            "a confirmed no-change result with unchanged valid actions and zero reward. "
            "Model, c28 serving, prompts, sampling, scorer, and game order remain unchanged. "
            "This is not a verified score improvement.\n\n"
        )
    intro["source"] = (heading + "".join(intro.get("source", []))).splitlines(keepends=True)

    cell9, source = _cell(notebook, "# Each bundled repo exposes its importable tree")
    if not control:
        patch_source = PATCH_SOURCE.read_text(encoding="utf-8")
        stage_anchor = "# Each bundled repo exposes its importable tree at <repo>/src or <repo>.\n"
        stage = patch_source + '''
_agent_source_path = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
if _agent_source_path.is_symlink() or not _agent_source_path.is_file():
    raise RuntimeError("Flash tool-agent source is missing or a symlink.")
_agent_original = _agent_source_path.read_bytes()
_agent_patched = patch_tool_agent(_agent_original)
FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT = WORKING_DIR / "flash_zero_change_repeat_guard_overlay"
_agent_overlay = FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT / "inference" / "agent"
_agent_overlay.mkdir(parents=True, exist_ok=True)
(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT / "inference" / "__init__.py").write_text(
    "from pkgutil import extend_path\\n__path__ = extend_path(__path__, __name__)\\n",
    encoding="utf-8",
)
(_agent_overlay / "__init__.py").write_text(
    "from pkgutil import extend_path\\n"
    "__path__ = extend_path(__path__, __name__)\\n"
    "from inference.agent.tool_agent import ToolAgent\\n"
    "from inference.agent.runtime_state import Frame, HistoryEntry\\n"
    "__all__ = [\\"ToolAgent\\", \\"Frame\\", \\"HistoryEntry\\"]\\n",
    encoding="utf-8",
)
_agent_overlay_path = _agent_overlay / "tool_agent.py"
_agent_overlay_path.write_text(_agent_patched, encoding="utf-8")
_zero_change_repeat_guard_patch_record = {
    "patch": PATCH_NAME,
    "source_sha256": hashlib.sha256(_agent_original).hexdigest(),
    "patched_sha256": hashlib.sha256(_agent_patched.encode("utf-8")).hexdigest(),
    "overlay_tool_agent_sha256": hashlib.sha256(_agent_overlay_path.read_bytes()).hexdigest(),
}
if (_zero_change_repeat_guard_patch_record["patched_sha256"]
        != _zero_change_repeat_guard_patch_record["overlay_tool_agent_sha256"]):
    raise RuntimeError("Flash zero-change guard overlay digest changed while staging.")
(WORKING_DIR / "flash_zero_change_repeat_guard_patch.json").write_text(
    json.dumps(_zero_change_repeat_guard_patch_record, indent=2) + "\\n", encoding="utf-8"
)
print(
    "FLASH_ZERO_CHANGE_REPEAT_GUARD_PATCH",
    json.dumps(_zero_change_repeat_guard_patch_record),
    flush=True,
)

'''
        source = _replace_once(source, stage_anchor, stage + stage_anchor, "patch staging")

        path_anchor = 'print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")\n'
        path_injection = path_anchor + '''if any(name == "inference" or name.startswith("inference.") for name in sys.modules):
    raise RuntimeError("Flash source loaded before the zero-change guard overlay could be installed.")
sys.path.insert(0, str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [
        str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT), os.environ.get("PYTHONPATH", "")
    ] if entry
)
print(
    "FLASH_ZERO_CHANGE_REPEAT_GUARD_OVERLAY_READY",
    json.dumps({
        "root": str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT),
        "tool_agent_sha256": _zero_change_repeat_guard_patch_record["overlay_tool_agent_sha256"],
    }),
    flush=True,
)
'''
        source = _replace_once(source, path_anchor, path_injection, "overlay path")

        finish_anchor = '''for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
    if entry not in sys.path:
        sys.path.insert(0, entry)'''
        finish = finish_anchor + '''

if any(name == "inference" or name.startswith("inference.") for name in sys.modules):
    raise RuntimeError("Flash source loaded during setup before the zero-change guard overlay could be used.")
if str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT) in sys.path:
    sys.path.remove(str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT))
sys.path.insert(0, str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [
        str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT),
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
          if item and item != str(FLASH_ZERO_CHANGE_REPEAT_GUARD_ROOT)),
    ] if entry
)
print("FLASH_ZERO_CHANGE_REPEAT_GUARD_OVERLAY_REASSERTED", flush=True)
import importlib

_agent_module = importlib.import_module("inference.agent.tool_agent")
_agent_module_path = Path(getattr(_agent_module, "__file__", "")).resolve()
if _agent_module_path != _agent_overlay_path.resolve():
    raise RuntimeError(f"Flash zero-change guard overlay was not imported: {_agent_module_path}.")
_agent_module_sha256 = hashlib.sha256(_agent_module_path.read_bytes()).hexdigest()
if _agent_module_sha256 != _zero_change_repeat_guard_patch_record["overlay_tool_agent_sha256"]:
    raise RuntimeError("Imported Flash zero-change guard overlay digest changed.")
print(
    "FLASH_ZERO_CHANGE_REPEAT_GUARD_OVERLAY_IMPORTED",
    json.dumps({
        "tool_agent_path": str(_agent_module_path),
        "tool_agent_sha256": _agent_module_sha256,
    }),
    flush=True,
)
'''
        source = _replace_once(source, finish_anchor, finish, "overlay import")

    cell9["source"] = source.splitlines(keepends=True)


def build(args: argparse.Namespace) -> Path:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    if args.mode != "full":
        raise ValueError("This bounded rerun builder only supports full 25-game packages.")
    _parse_kernel_id(args.kernel_id)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    patch_notebook(notebook, control=args.control)
    _clean_execution(notebook)
    notebook_name = "arc-agi3-qwen38-flash-next-mtp-full.ipynb"
    title = args.title or _parse_kernel_id(args.kernel_id)[1].replace("-", " ")
    metadata = _metadata(args.kernel_id, notebook_name, title)
    compiled = _compile_notebook(notebook, "pulled-zero-change-guard-full")

    output = (args.output_dir or BUILD_ROOT / "pulled-flash-zero-change-guard-full").resolve()
    if output == REPO_ROOT or REPO_ROOT not in output.parents:
        raise ValueError("--output-dir must stay inside this repository.")
    if output.exists():
        marker = output / PACKAGE_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "owned":
            raise ValueError(f"Refusing to replace unowned output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / PACKAGE_MARKER).write_text("owned\n", encoding="utf-8")
    (output / notebook_name).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _compile_notebook(json.loads((output / notebook_name).read_text()), str(output / notebook_name))
    print(f"Built pulled Flash {'control' if args.control else 'zero-change guard'} package: {output}")
    print(f"Compiled code cells: {compiled}")
    print("No Kaggle upload, run, or submission was performed.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
