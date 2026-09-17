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
PUBLIC_GAME_COUNT = 25
PUBLIC_GAME_IDS = (
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940",
    "wa30-ee6fef47", "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b",
    "sp80-589a99af", "m0r0-492f87ba", "vc33-5430563c", "ar25-0c556536",
    "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
    "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab",
    "tr87-cd924810",
)
MAX_TERMINAL_GRACE_SECONDS = 599


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


def _validate_preflight_setting(value: int | None, *, name: str, maximum: int | None = None) -> None:
    if value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


def _validate_terminal_grace_seconds(value: int | None) -> None:
    if value is None:
        return
    if type(value) is not int or not 0 <= value <= MAX_TERMINAL_GRACE_SECONDS:
        raise ValueError(
            "terminal_grace_seconds must be an integer from 0 to "
            f"{MAX_TERMINAL_GRACE_SECONDS}."
        )


def _add_preflight_support(
    notebook: dict,
    *,
    max_games: int | None,
    concurrency: int | None = None,
    game_id: str | None = None,
    runtime_seconds: int | None = None,
    terminal_grace_seconds: int | None = None,
    analyzer_timeout: int | None = None,
    include_agent_state_patch: bool = False,
) -> None:
    _validate_preflight_setting(max_games, name="max_games", maximum=PUBLIC_GAME_COUNT)
    _validate_preflight_setting(concurrency, name="concurrency")
    _validate_preflight_setting(runtime_seconds, name="runtime_seconds")
    _validate_preflight_setting(analyzer_timeout, name="analyzer_timeout")
    _validate_terminal_grace_seconds(terminal_grace_seconds)
    if game_id is not None and game_id not in PUBLIC_GAME_IDS:
        raise ValueError(f"game_id must be one of the pinned public game IDs: {game_id!r}.")
    if game_id is not None and max_games not in (None, 1):
        raise ValueError("game_id preflights must select exactly one game.")

    start_anchor = "NOTEBOOK_START_EPOCH = time.time()\n"
    start_injection = (
        "\n"
        "FLASH_OFFLINE_MAX_GAMES = int(os.environ.get(\"FLASH_OFFLINE_MAX_GAMES\", \"0\"))\n"
        "FLASH_OFFLINE_GAME_ID = os.environ.get(\"FLASH_OFFLINE_GAME_ID\", \"\").strip()\n"
        "FLASH_RUNTIME_CONCURRENCY = int(os.environ.get(\"FLASH_RUNTIME_CONCURRENCY\", \"0\"))\n"
        "FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get(\"FLASH_OFFLINE_MAX_RUNTIME_S\", \"0\"))\n"
        "FLASH_TERMINAL_GRACE_S = float(os.environ.get(\"FLASH_TERMINAL_GRACE_S\", \"0\"))\n"
        "FLASH_ANALYZER_TIMEOUT = float(os.environ.get(\"FLASH_ANALYZER_TIMEOUT\", \"900\"))\n"
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
        "if not TRUE_SUBMISSION and FLASH_OFFLINE_GAME_ID:\n"
        "    if FLASH_OFFLINE_GAME_ID not in PUBLIC_GAME_IDS:\n"
        "        raise RuntimeError(f'Unknown offline game ID: {FLASH_OFFLINE_GAME_ID!r}.')\n"
        "    PUBLIC_GAME_IDS = tuple(game_id for game_id in PUBLIC_GAME_IDS\n"
        "                            if game_id == FLASH_OFFLINE_GAME_ID)\n\n"
        "elif not TRUE_SUBMISSION and FLASH_OFFLINE_MAX_GAMES:\n"
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
                "    extra = [] if (not TRUE_SUBMISSION and (FLASH_OFFLINE_MAX_GAMES or FLASH_OFFLINE_GAME_ID)) else sorted(set(offline_by_id) - set(PUBLIC_GAME_IDS))\n",
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
    teardown_patch_source = (REPO_ROOT / "scripts/flash_teardown_patch.py").read_text(encoding="utf-8")
    agent_patch_source = (
        (REPO_ROOT / "scripts/flash_agent_state_patch.py").read_text(encoding="utf-8")
        if include_agent_state_patch else ""
    )
    stage_source = teardown_patch_source + ("\n" + agent_patch_source if agent_patch_source else "") + """
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

""" + ("""
_agent_source_path = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
if _agent_source_path.is_symlink() or not _agent_source_path.is_file():
    raise RuntimeError("Flash tool-agent source is missing or a symlink.")
_agent_original = _agent_source_path.read_bytes()
_agent_patched = patch_tool_agent(_agent_original)
FLASH_AGENT_OVERLAY_ROOT = WORKING_DIR / "flash_agent_overlay"
_agent_overlay = FLASH_AGENT_OVERLAY_ROOT / "inference" / "agent"
_agent_overlay.mkdir(parents=True, exist_ok=True)
(FLASH_AGENT_OVERLAY_ROOT / "inference" / "__init__.py").write_text(
    "from pkgutil import extend_path\\n__path__ = extend_path(__path__, __name__)\\n",
    encoding="utf-8",
)
(_agent_overlay / "__init__.py").write_text(
    "from pkgutil import extend_path\\n"
    "__path__ = extend_path(__path__, __name__)\\n"
    "from inference.agent.tool_agent import ToolAgent\\n"
    "from inference.agent.runtime_state import Frame, HistoryEntry\\n"
    "__all__ = [\\\"ToolAgent\\\", \\\"Frame\\\", \\\"HistoryEntry\\\"]\\n",
    encoding="utf-8",
)
_agent_overlay_path = _agent_overlay / "tool_agent.py"
_agent_overlay_path.write_text(_agent_patched, encoding="utf-8")
_agent_patch_record = {
    "patch": PATCH_NAME,
    "source_sha256": hashlib.sha256(_agent_original).hexdigest(),
    "patched_sha256": hashlib.sha256(_agent_patched.encode("utf-8")).hexdigest(),
    "overlay_tool_agent_sha256": hashlib.sha256(_agent_overlay_path.read_bytes()).hexdigest(),
}
if _agent_patch_record["patched_sha256"] != _agent_patch_record["overlay_tool_agent_sha256"]:
    raise RuntimeError("Flash agent-state overlay digest changed while staging.")
(WORKING_DIR / "flash_agent_state_patch.json").write_text(
    json.dumps(_agent_patch_record, indent=2) + "\\n", encoding="utf-8"
)
print("FLASH_AGENT_STATE_PATCH", json.dumps(_agent_patch_record), flush=True)
""" if include_agent_state_patch else "") + """

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

    source_path_anchor = "print(f\"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)\")\n"
    source_path_injection = source_path_anchor + """if any(name == "inference" or name.startswith("inference.") for name in sys.modules):
    raise RuntimeError("Flash source loaded before the agent-state overlay could be installed.")
sys.path.insert(0, str(FLASH_AGENT_OVERLAY_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [str(FLASH_AGENT_OVERLAY_ROOT), os.environ.get("PYTHONPATH", "")]
    if entry
)
print(
    "FLASH_AGENT_OVERLAY_READY",
    json.dumps({
        "root": str(FLASH_AGENT_OVERLAY_ROOT),
        "tool_agent_sha256": _agent_patch_record["overlay_tool_agent_sha256"],
    }),
    flush=True,
)
"""
    if not include_agent_state_patch:
        source_path_injection = source_path_anchor
    source_path_matches = 0
    for candidate in notebook["cells"]:
        if candidate.get("cell_type") != "code":
            continue
        candidate_source = "".join(candidate.get("source", []))
        source_path_matches += candidate_source.count(source_path_anchor)
        candidate_source = candidate_source.replace(source_path_anchor, source_path_injection)
        candidate["source"] = candidate_source.splitlines(keepends=True)
    if source_path_matches != 1:
        raise ValueError(
            "Flash agent-state overlay anchor did not match exactly once: "
            f"source_path={source_path_matches}."
        )

    setup_finish_anchor = """for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
    if entry not in sys.path:
        sys.path.insert(0, entry)"""
    setup_finish_injection = setup_finish_anchor + """

if any(name == "inference" or name.startswith("inference.") for name in sys.modules):
    raise RuntimeError("Flash source loaded during setup before the agent-state overlay could be used.")
if str(FLASH_AGENT_OVERLAY_ROOT) in sys.path:
    sys.path.remove(str(FLASH_AGENT_OVERLAY_ROOT))
sys.path.insert(0, str(FLASH_AGENT_OVERLAY_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    entry for entry in [
        str(FLASH_AGENT_OVERLAY_ROOT),
        *(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
          if item and item != str(FLASH_AGENT_OVERLAY_ROOT)),
    ] if entry
)
print("FLASH_AGENT_OVERLAY_REASSERTED", flush=True)
import importlib

_agent_module = importlib.import_module("inference.agent.tool_agent")
_agent_module_path = Path(getattr(_agent_module, "__file__", "")).resolve()
if _agent_module_path != _agent_overlay_path.resolve():
    raise RuntimeError(
        f"Flash agent-state overlay was not imported: {_agent_module_path}."
    )
_agent_module_sha256 = hashlib.sha256(_agent_module_path.read_bytes()).hexdigest()
if _agent_module_sha256 != _agent_patch_record["overlay_tool_agent_sha256"]:
    raise RuntimeError("Imported Flash agent-state overlay digest changed.")
print(
    "FLASH_AGENT_OVERLAY_IMPORTED",
    json.dumps({
        "tool_agent_path": str(_agent_module_path),
        "tool_agent_sha256": _agent_module_sha256,
    }),
    flush=True,
)
"""
    if not include_agent_state_patch:
        setup_finish_injection = setup_finish_anchor
    setup_finish_matches = 0
    for candidate in notebook["cells"]:
        if candidate.get("cell_type") != "code":
            continue
        candidate_source = "".join(candidate.get("source", []))
        setup_finish_matches += candidate_source.count(setup_finish_anchor)
        candidate_source = candidate_source.replace(setup_finish_anchor, setup_finish_injection)
        candidate["source"] = candidate_source.splitlines(keepends=True)
    if setup_finish_matches != 1:
        raise ValueError(
            "Flash agent-state setup-finish anchor did not match exactly once: "
            f"setup_finish={setup_finish_matches}."
        )

    if max_games is not None or runtime_seconds is not None:
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "FLASH_OFFLINE_MAX_GAMES" in "".join(cell.get("source", []))
        )
        source = "".join(cell["source"])
        runtime_default = 1800 if runtime_seconds is None else runtime_seconds
        source = source.replace(
            'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "0"))',
            f'FLASH_OFFLINE_MAX_GAMES = int(os.environ.get("FLASH_OFFLINE_MAX_GAMES", "{max_games or 0}"))',
            1,
        )
        source = source.replace(
            'FLASH_OFFLINE_MAX_RUNTIME_S = float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "0"))',
            f'FLASH_OFFLINE_MAX_RUNTIME_S = 0.0 if TRUE_SUBMISSION else float(os.environ.get("FLASH_OFFLINE_MAX_RUNTIME_S", "{runtime_default}"))',
            1,
        )
        grace_default = 120 if terminal_grace_seconds is None else terminal_grace_seconds
        source = source.replace(
            'FLASH_TERMINAL_GRACE_S = float(os.environ.get("FLASH_TERMINAL_GRACE_S", "0"))',
            f'FLASH_TERMINAL_GRACE_S = 0.0 if TRUE_SUBMISSION else float(os.environ.get("FLASH_TERMINAL_GRACE_S", "{grace_default}"))',
            1,
        )
        if game_id is not None:
            source = source.replace(
                'FLASH_OFFLINE_GAME_ID = os.environ.get("FLASH_OFFLINE_GAME_ID", "").strip()',
                f'FLASH_OFFLINE_GAME_ID = os.environ.get("FLASH_OFFLINE_GAME_ID", "{game_id}").strip()',
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
                    'offline_batch_count = max(1, (len(bm.games) + int(bm.solver.concurrency) - 1) '
                    '// int(bm.solver.concurrency))\n'
                    'budget = min(float(getattr(target, "max_runtime_s", 0.0) or 0.0), '
                    'FLASH_OFFLINE_MAX_RUNTIME_S * offline_batch_count + 600.0) '
                    'if FLASH_OFFLINE_MAX_RUNTIME_S > 0 else '
                    'float(getattr(target, "max_runtime_s", 0.0) or 0.0)',
                )
            candidate["source"] = candidate_source.splitlines(keepends=True)
        if replaced_settings != 1 or replaced_budget != 1:
            raise ValueError(
                "Flash preflight runtime anchors did not match exactly once: "
                f"settings={replaced_settings}, budget={replaced_budget}."
            )

        deadline_anchor = """    ),
)

# Play the benchmark; watchdog stop and teardown run even if it raises.
"""
        deadline_replacement = """    ),
)

# Bounded offline checks measure gameplay after serving setup. The small grace
# lets the solver's own per-game cap finalize `gave_up` before the global
# notebook deadline would mark the game cancelled.
if not TRUE_SUBMISSION and FLASH_OFFLINE_MAX_RUNTIME_S > 0:
    if not 0.0 <= FLASH_TERMINAL_GRACE_S < 600.0:
        raise RuntimeError("FLASH_TERMINAL_GRACE_S must stay within the teardown reserve.")
    gameplay_budget_s = budget - 600.0
    soft_end = datetime.now() + timedelta(
        seconds=gameplay_budget_s + FLASH_TERMINAL_GRACE_S
    )
    print(
        f\"PUBLIC25_DEADLINE origin=post_setup gameplay_budget_s={gameplay_budget_s} \"
        f\"terminal_grace_s={FLASH_TERMINAL_GRACE_S} \"
        f\"offline_batch_count={offline_batch_count}\",
        flush=True,
    )
elif not TRUE_SUBMISSION:
    print(
        f\"PUBLIC25_DEADLINE origin=notebook_start gameplay_budget_s={budget - 600.0}\",
        flush=True,
    )

# Play the benchmark; watchdog stop and teardown run even if it raises.
"""
        deadline_matches = 0
        for candidate in notebook["cells"]:
            if candidate.get("cell_type") != "code":
                continue
            candidate_source = "".join(candidate.get("source", []))
            deadline_matches += candidate_source.count(deadline_anchor)
            candidate_source = candidate_source.replace(deadline_anchor, deadline_replacement)
            candidate["source"] = candidate_source.splitlines(keepends=True)
        if deadline_matches != 1:
            raise ValueError(
                "Flash preflight deadline anchor did not match exactly once: "
                f"deadline={deadline_matches}."
            )

    if analyzer_timeout is not None:
        settings_anchor = "bm.solver.analyzer_timeout = 900.0"
        replaced_timeout = 0
        for candidate in notebook["cells"]:
            if candidate.get("cell_type") != "code":
                continue
            candidate_source = "".join(candidate.get("source", []))
            if settings_anchor in candidate_source:
                replaced_timeout += candidate_source.count(settings_anchor)
                candidate_source = candidate_source.replace(
                    settings_anchor,
                    "bm.solver.analyzer_timeout = (FLASH_ANALYZER_TIMEOUT "
                    "if FLASH_ANALYZER_TIMEOUT > 0 else 900.0)",
                )
            candidate["source"] = candidate_source.splitlines(keepends=True)
        if replaced_timeout != 1:
            raise ValueError(
                "Flash analyzer timeout anchor did not match exactly once: "
                f"analyzer_timeout={replaced_timeout}."
            )
        cell = next(
            cell for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "FLASH_ANALYZER_TIMEOUT" in "".join(cell.get("source", []))
        )
        source = "".join(cell["source"])
        source = source.replace(
            'FLASH_ANALYZER_TIMEOUT = float(os.environ.get("FLASH_ANALYZER_TIMEOUT", "900"))',
            f'FLASH_ANALYZER_TIMEOUT = float(os.environ.get("FLASH_ANALYZER_TIMEOUT", "{analyzer_timeout}"))',
            1,
        )
        cell["source"] = source.splitlines(keepends=True)

    if concurrency is not None:
        concurrency_anchor = "bm.solver.concurrency = 28"
        replaced_concurrency = 0
        for candidate in notebook["cells"]:
            if candidate.get("cell_type") != "code":
                continue
            candidate_source = "".join(candidate.get("source", []))
            if concurrency_anchor in candidate_source:
                replaced_concurrency += candidate_source.count(concurrency_anchor)
                candidate_source = candidate_source.replace(
                    concurrency_anchor,
                    "bm.solver.concurrency = (FLASH_RUNTIME_CONCURRENCY "
                    "if FLASH_RUNTIME_CONCURRENCY > 0 else 28)",
                )
            candidate["source"] = candidate_source.splitlines(keepends=True)
        if replaced_concurrency != 1:
            raise ValueError(
                "Flash runtime concurrency anchor did not match exactly once: "
                f"concurrency={replaced_concurrency}."
            )
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "FLASH_RUNTIME_CONCURRENCY" in "".join(cell.get("source", []))
        )
        source = "".join(cell["source"])
        source = source.replace(
            'FLASH_RUNTIME_CONCURRENCY = int(os.environ.get("FLASH_RUNTIME_CONCURRENCY", "0"))',
            f'FLASH_RUNTIME_CONCURRENCY = int(os.environ.get("FLASH_RUNTIME_CONCURRENCY", "{concurrency}"))',
            1,
        )
        cell["source"] = source.splitlines(keepends=True)


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
    if args.mode == "full" and any(
        value is not None
        for value in (
            args.max_games,
            args.game_id,
            args.runtime_seconds,
            args.terminal_grace_seconds,
        )
    ):
        raise ValueError(
            "--max-games, --game-id, --runtime-seconds, and --terminal-grace-seconds "
            "are only valid for preflight packages."
        )
    if args.game_id is not None and args.max_games not in (None, 1):
        raise ValueError("--game-id can only be combined with --max-games 1.")
    max_games = 1 if args.mode == "preflight" and args.max_games is None else args.max_games
    concurrency = 28 if args.mode == "preflight" and args.concurrency is None else args.concurrency
    runtime_seconds = (
        1800 if args.mode == "preflight" and args.runtime_seconds is None
        else args.runtime_seconds
    )
    terminal_grace_seconds = (
        120 if args.mode == "preflight" and args.terminal_grace_seconds is None
        else args.terminal_grace_seconds
    )
    _add_preflight_support(
        notebook,
        max_games=max_games,
        concurrency=concurrency,
        game_id=args.game_id,
        runtime_seconds=runtime_seconds,
        terminal_grace_seconds=terminal_grace_seconds,
        analyzer_timeout=getattr(args, "analyzer_timeout", None),
        include_agent_state_patch=getattr(args, "agent_state_patch", False),
    )
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
    parser.add_argument(
        "--max-games",
        type=int,
        help="Preflight offline game limit (default: 1; full packages always use all 25 games).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Measured Duck concurrency override (preflight and full; default: 28).",
    )
    parser.add_argument(
        "--game-id",
        help="Run one pinned offline public game (preflight only).",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=int,
        help="Per-game offline runtime for an isolated preflight (default: 1800).",
    )
    parser.add_argument(
        "--terminal-grace-seconds",
        type=int,
        help="Bounded post-runtime grace for clean preflight termination (default: 120).",
    )
    parser.add_argument(
        "--analyzer-timeout",
        type=int,
        help="Analyzer request timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--agent-state-patch",
        dest="agent_state_patch",
        action="store_true",
        help="Include the experimental reasoning-state overlay (off by default).",
    )
    parser.set_defaults(agent_state_patch=False)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
