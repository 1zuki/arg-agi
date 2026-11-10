#!/usr/bin/env python3
"""Build an opt-in candidate from the downloaded 3.40 Flash notebook."""

from __future__ import annotations

import argparse
import ast
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
DIRECT_RETRY_PATCH = REPO_ROOT / "scripts/flash_direct_tool_retry_patch.py"


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


def patch_notebook(
    notebook: dict,
    *,
    mode: str,
    max_games: int | None,
    runtime_seconds: int | None,
    terminal_grace_seconds: int,
) -> None:
    if mode not in {"preflight", "full"}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if mode == "preflight":
        if max_games is None or not 1 <= max_games <= 25:
            raise ValueError("preflight max_games must be between 1 and 25.")
        if runtime_seconds is None or runtime_seconds < 1:
            raise ValueError("preflight runtime_seconds must be positive.")
        if not 0 <= terminal_grace_seconds <= 599:
            raise ValueError("terminal_grace_seconds must be between 0 and 599.")
    elif max_games is not None or runtime_seconds is not None:
        raise ValueError("max_games/runtime_seconds are preflight-only.")

    intro = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown")
    intro["source"] = (
        "## Experimental direct-tool retry candidate\n\n"
        "Based on the downloaded 3.40 Flash notebook. The only solver-policy change is a "
        "digest-bound `tool_agent.py` patch that makes tool-less responses retry with a "
        "direct Python tool call. Model, c28 serving, prompts, sampling, actions, and scorer "
        "remain unchanged. This is not a verified score improvement.\n\n"
        + "".join(intro.get("source", []))
    ).splitlines(keepends=True)

    cell3, source = _cell(notebook, "FLASH_OFFLINE_MAX_GAMES =")
    if mode == "preflight":
        source = _replace_once(
            source,
            'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "0"))',
            f'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "{max_games}"))',
            "game default",
        )
        source = _replace_once(
            source,
            'FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "0"))',
            f'FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "{runtime_seconds}"))',
            "runtime default",
        )
    cell3["source"] = source.splitlines(keepends=True)

    cell9, source = _cell(notebook, "# Each bundled repo exposes its importable tree")
    patch_source = DIRECT_RETRY_PATCH.read_text(encoding="utf-8")
    stage_anchor = "# Each bundled repo exposes its importable tree at <repo>/src or <repo>.\n"
    stage = patch_source + '''
_agent_source_path = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
if _agent_source_path.is_symlink() or not _agent_source_path.is_file():
    raise RuntimeError("Flash tool-agent source is missing or a symlink.")
_agent_original = _agent_source_path.read_bytes()
_agent_patched = patch_tool_agent(_agent_original)
FLASH_DIRECT_RETRY_ROOT = WORKING_DIR / "flash_direct_retry_overlay"
_agent_overlay = FLASH_DIRECT_RETRY_ROOT / "inference" / "agent"
_agent_overlay.mkdir(parents=True, exist_ok=True)
(FLASH_DIRECT_RETRY_ROOT / "inference" / "__init__.py").write_text(
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
_direct_retry_patch_record = {
    "patch": PATCH_NAME,
    "source_sha256": hashlib.sha256(_agent_original).hexdigest(),
    "patched_sha256": hashlib.sha256(_agent_patched.encode("utf-8")).hexdigest(),
    "overlay_tool_agent_sha256": hashlib.sha256(_agent_overlay_path.read_bytes()).hexdigest(),
}
if _direct_retry_patch_record["patched_sha256"] != _direct_retry_patch_record["overlay_tool_agent_sha256"]:
    raise RuntimeError("Flash direct-retry overlay digest changed while staging.")
(WORKING_DIR / "flash_direct_retry_patch.json").write_text(
    json.dumps(_direct_retry_patch_record, indent=2) + "\\n", encoding="utf-8"
)
print("FLASH_DIRECT_RETRY_PATCH", json.dumps(_direct_retry_patch_record), flush=True)

'''
    source = _replace_once(source, stage_anchor, stage + stage_anchor, "patch staging")
    path_anchor = 'print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")\n'
    path_injection = path_anchor + '''if any(name == "inference" or name.startswith("inference.") for name in sys.modules):
    raise RuntimeError("Flash source loaded before the direct-retry overlay could be installed.")
sys.path.insert(0, str(FLASH_DIRECT_RETRY_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [str(FLASH_DIRECT_RETRY_ROOT), os.environ.get("PYTHONPATH", "")]
    if entry
)
print(
    "FLASH_DIRECT_RETRY_OVERLAY_READY",
    json.dumps({
        "root": str(FLASH_DIRECT_RETRY_ROOT),
        "tool_agent_sha256": _direct_retry_patch_record["overlay_tool_agent_sha256"],
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
    raise RuntimeError("Flash source loaded during setup before the direct-retry overlay could be used.")
if str(FLASH_DIRECT_RETRY_ROOT) in sys.path:
    sys.path.remove(str(FLASH_DIRECT_RETRY_ROOT))
sys.path.insert(0, str(FLASH_DIRECT_RETRY_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [
        str(FLASH_DIRECT_RETRY_ROOT),
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
          if item and item != str(FLASH_DIRECT_RETRY_ROOT)),
    ] if entry
)
print("FLASH_DIRECT_RETRY_OVERLAY_REASSERTED", flush=True)
import importlib

_agent_module = importlib.import_module("inference.agent.tool_agent")
_agent_module_path = Path(getattr(_agent_module, "__file__", "")).resolve()
if _agent_module_path != _agent_overlay_path.resolve():
    raise RuntimeError(f"Flash direct-retry overlay was not imported: {_agent_module_path}.")
_agent_module_sha256 = hashlib.sha256(_agent_module_path.read_bytes()).hexdigest()
if _agent_module_sha256 != _direct_retry_patch_record["overlay_tool_agent_sha256"]:
    raise RuntimeError("Imported Flash direct-retry overlay digest changed.")
print(
    "FLASH_DIRECT_RETRY_OVERLAY_IMPORTED",
    json.dumps({
        "tool_agent_path": str(_agent_module_path),
        "tool_agent_sha256": _agent_module_sha256,
    }),
    flush=True,
)
'''
    source = _replace_once(source, finish_anchor, finish, "overlay import")
    cell9["source"] = source.splitlines(keepends=True)

    if mode == "preflight":
        cell13, source = _cell(notebook, "bm.solver.max_runtime_s_per_game = 7920.0")
        source = _replace_once(
            source,
            "bm.solver.max_runtime_s_per_game = 7920.0",
            "bm.solver.max_runtime_s_per_game = (FLASH_OFFLINE_MAX_RUNTIME_S if FLASH_OFFLINE_MAX_RUNTIME_S > 0 else 7920.0)",
            "preflight runtime",
        )
        cell13["source"] = source.splitlines(keepends=True)

        cell15, source = _cell(notebook, "# Outside a real submission, stop ~10 min before")
        old = '''# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
if budget <= 600.0:
    raise RuntimeError(f'Notebook budget is too small for the teardown reserve: {budget}.')
soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)
'''
        new = f'''# Candidate preflight measures gameplay after serving setup.
_budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
if _budget <= 600.0:
    raise RuntimeError(f'Notebook budget is too small for the teardown reserve: {{_budget}}.')
_candidate_batches = max(1, (len(PUBLIC_GAME_IDS) + int(bm.solver.concurrency) - 1) // int(bm.solver.concurrency))
_candidate_gameplay_budget_s = FLASH_OFFLINE_MAX_RUNTIME_S * _candidate_batches
soft_end = datetime.now() + timedelta(seconds=_candidate_gameplay_budget_s + {terminal_grace_seconds}.0)
print(
    f"FLASH_CANDIDATE_DEADLINE origin=post_setup gameplay_budget_s={{_candidate_gameplay_budget_s}} "
    f"terminal_grace_s={terminal_grace_seconds}.0 batches={{_candidate_batches}}",
    flush=True,
)
'''
        source = _replace_once(source, old, new, "preflight deadline")
        cell15["source"] = source.splitlines(keepends=True)


def build(args: argparse.Namespace) -> Path:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    _parse_kernel_id(args.kernel_id)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    patch_notebook(
        notebook,
        mode=args.mode,
        max_games=args.max_games,
        runtime_seconds=args.runtime_seconds,
        terminal_grace_seconds=args.terminal_grace_seconds,
    )
    _clean_execution(notebook)
    notebook_name = f"arc-agi3-qwen38-flash-next-mtp-{args.mode}.ipynb"
    title = args.title or _parse_kernel_id(args.kernel_id)[1].replace("-", " ")
    metadata = _metadata(args.kernel_id, notebook_name, title)
    compiled = _compile_notebook(notebook, f"pulled-{args.mode}")

    output = (args.output_dir or BUILD_ROOT / f"pulled-flash-direct-retry-{args.mode}").resolve()
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
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _compile_notebook(json.loads(notebook_path.read_text()), str(notebook_path))
    print(f"Built pulled Flash candidate: {output}")
    print(f"Compiled code cells: {compiled}")
    print("No Kaggle upload, run, or submission was performed.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--runtime-seconds", type=int)
    parser.add_argument("--terminal-grace-seconds", type=int, default=120)
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
