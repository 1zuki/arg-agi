from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'build/flash-baseline-source-20260920/tool_agent.py'
PATCH_PATH = ROOT / 'scripts/flash_direct_tool_retry_patch.py'
sys.path.insert(0, str(ROOT / 'scripts'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectToolRetryTests(unittest.TestCase):
    def test_patch_is_digest_bound_and_changes_only_followup_prefix(self):
        if not SOURCE.is_file():
            self.skipTest('pinned source fixture not downloaded')
        patch = load(PATCH_PATH, 'direct_retry')
        source = SOURCE.read_bytes()
        patched = patch.patch_tool_agent(source)
        self.assertEqual(hashlib.sha256(source).hexdigest(), patch.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(patched.encode()).hexdigest(),
                         '89a042e1a7824db7c48591d41a35d5e0d45e0e4842a4817abac7f6b5fa98fe53')
        self.assertIn('You did not call a tool. On this retry, emit exactly one `python` tool call directly', patched)
        self.assertEqual(patched.count('You did not call a tool. On this retry'), 1)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            patch.patch_tool_agent(source + b'\n')

    def test_candidate_builds_from_pulled_control_and_is_not_action_or_state_patch(self):
        builder = load(ROOT / 'scripts/build_pulled_flash_candidate.py', 'pulled_builder')
        notebook = json.loads(builder.SOURCE_NOTEBOOK.read_text())
        builder.patch_notebook(notebook, mode='preflight', max_games=25,
                               runtime_seconds=1800, terminal_grace_seconds=120)
        builder._compile_notebook(notebook, 'direct-retry-test')
        source = '\n'.join(''.join(c.get('source', [])) for c in notebook['cells']
                           if c.get('cell_type') == 'code')
        self.assertIn('FLASH_DIRECT_RETRY_PATCH', source)
        self.assertIn('FLASH_DIRECT_RETRY_OVERLAY_IMPORTED', source)
        self.assertNotIn('FLASH_ACTION7_PATCH', source)
        self.assertNotIn('FLASH_AGENT_STATE_PATCH', source)
        self.assertIn('FLASH_CANDIDATE_DEADLINE', source)
        self.assertIn('not a verified score improvement', ''.join(notebook['cells'][0]['source']).lower())

    def test_generated_overlay_imports_exact_patched_module(self):
        builder = load(ROOT / 'scripts/build_pulled_flash_candidate.py', 'pulled_builder_import')
        notebook = json.loads(builder.SOURCE_NOTEBOOK.read_text())
        builder.patch_notebook(notebook, mode='preflight', max_games=1,
                               runtime_seconds=1800, terminal_grace_seconds=120)
        source = ''.join(notebook['cells'][9]['source'])
        self.assertIn('FLASH_DIRECT_RETRY_ROOT', source)
        self.assertIn('FLASH_DIRECT_RETRY_OVERLAY_IMPORTED', source)
        self.assertIn('importlib.import_module("inference.agent.tool_agent")', source)


if __name__ == '__main__':
    unittest.main()
