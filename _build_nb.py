import json, os

here = os.path.dirname(os.path.abspath(__file__))


def _strip_module_imports(src):
    # Drop the leading import block from model.py so it can be inlined
    # into the agent cell without duplicate/ordering issues; the agent
    # file already imports numpy/torch/F/nn. Keep everything from the
    # first non-import, non-comment, non-blank line onward.
    lines = src.splitlines(keepends=True)
    start = 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st == "" or st.startswith("#"):
            continue
        if st.startswith("import ") or st.startswith("from "):
            start = i + 1
            continue
        break
    return "".join(lines[start:])


def _build_agent_src():
    # On Kaggle there is no separate cwm.py, so inline it into the agent
    # source and strip the `from cwm import ...` line (marked with
    # NB_STRIP_IMPORT). Single source of truth stays cwm.py.
    model_src = open(os.path.join(here, "cwm.py"), encoding="utf-8").read()
    agent_src = open(os.path.join(here, "my_agent.py"), encoding="utf-8").read()
    agent_lines = [ln for ln in agent_src.splitlines(keepends=True)
                   if "NB_STRIP_IMPORT" not in ln]
    agent_src = "".join(agent_lines)
    model_inline = _strip_module_imports(model_src)
    banner = "# ---- inlined from cwm.py (single source of truth) ----\n"
    # place the inlined model right after the agent's import block:
    # find the last top-level import line in the agent
    out, inserted = [], False
    al = agent_src.splitlines(keepends=True)
    last_imp = 0
    for i, ln in enumerate(al):
        # only top-level (column-0) imports; an indented `import json`
        # inside a method must NOT count as the insertion point.
        if ln.startswith("import ") or ln.startswith("from "):
            last_imp = i
    for i, ln in enumerate(al):
        out.append(ln)
        if i == last_imp and not inserted:
            out.append("\n" + banner + model_inline + "\n")
            inserted = True
    return "".join(out)


agent_src = _build_agent_src()

def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

# Cell 0: install arc-agi wheels offline (matches competition wheel path)
c0 = code_cell(
    '!pip install --no-index --find-links \\\n'
    '    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n'
    '    arc-agi python-dotenv'
)

# Cell 1: write the agent to /kaggle/working/my_agent.py
# (writefile magic must be the first line of the cell)
c1 = code_cell("%%writefile /kaggle/working/my_agent.py\n" + agent_src)

# Cell 2: on competition rerun, wire the agent into the harness and run it
c2 = code_cell(
    "import os\n"
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games\n"
    "    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents\n"
    "    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py\n"
    "    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py','w') as f:\n"
    "        f.write(\"\"\"from typing import Type\n"
    "from dotenv import load_dotenv\n"
    "from .agent import Agent, Playback\n"
    "from .swarm import Swarm\n"
    "from .templates.random_agent import Random\n"
    "from .templates.my_agent import MyAgent\n"
    "load_dotenv()\n"
    "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\"random\": Random, \"myagent\": MyAgent}\n"
    "\"\"\")\n"
    "    with open('/kaggle/working/ARC-AGI-3-Agents/.env','w') as f:\n"
    "        f.write(\"\"\"SCHEME=http\n"
    "HOST=gateway\n"
    "PORT=8001\n"
    "ARC_API_KEY=test-key-123\n"
    "ARC_BASE_URL=http://gateway:8001/\n"
    "OPERATION_MODE=online\n"
    "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    "\"\"\")\n"
    "    os.environ.setdefault('ARC_PLANNER', 'bfs')\n"
    "    os.environ.setdefault('ARC_BFS_DEPTH', '3')\n"
    "    os.environ.setdefault('ARC_BFS_BEAM', '8')\n"
    "    os.environ.setdefault('ARC_MCTS_DEPTH', '3')\n"
    "    os.environ.setdefault('ARC_MCTS_SIMS', '32')\n"
    "    os.environ.setdefault('ARC_MCTS_CPUCT', '1.4')\n"
    "    os.environ.setdefault('ARC_CWM', '/kaggle/input/forge-pretrained-weights/cwm.pt')\n"
    "    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent"
)

# Cell 3: local-validation submission stub (only when NOT a competition rerun)
c3 = code_cell(
    "import os\n"
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
    "    import pandas as pd\n"
    "    submission = pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])\n"
    "    submission.to_parquet('/kaggle/working/submission.parquet',index=False)"
)

nb = {
    "cells": [c0, c1, c2, c3],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(here, "arc-agi.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)

# validate
try:
    import nbformat
except ModuleNotFoundError:
    print("nbformat not installed; skipped nbformat validation")
else:
    nbformat.read(out, as_version=4)
    print("nbformat validation OK")
