from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/flash_action_names.py'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCH = load_script('flash_action7_patch')
BUILDER = load_script('build_flash_next_package')
AUDIT = load_script('audit_flash_output')


def generated_notebook(**kwargs):
    notebook = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
    BUILDER._add_preflight_support(notebook, **kwargs)
    BUILDER._compile_notebook(notebook, 'action7-test')
    return notebook


def code_source(notebook):
    return '\n'.join(''.join(c.get('source', [])) for c in notebook['cells']
                     if c.get('cell_type') == 'code')


class Action7PatchTests(unittest.TestCase):
    def test_real_pinned_mapping_regression_and_only_action7_changes(self):
        source = FIXTURE.read_bytes()
        before, after = {}, {}
        exec(compile(source, str(FIXTURE), 'exec'), before)
        patched = PATCH.patch_action_names(source)
        exec(compile(patched, 'patched-action-names', 'exec'), after)
        self.assertEqual(hashlib.sha256(source).hexdigest(), AUDIT.ACTION7_SOURCE_HASH)
        self.assertEqual(hashlib.sha256(patched.encode()).hexdigest(), AUDIT.ACTION7_PATCH_HASH)
        self.assertEqual(PATCH.PATCH_NAME, AUDIT.ACTION7_PATCH_NAME)
        self.assertEqual(before['to_model_action']('ACTION7'), 'ACTION7')
        self.assertIsNone(before['to_engine_action']('ACTION7'))
        self.assertEqual(after['to_engine_action'](' action7 '), 'ACTION7')
        self.assertEqual(after['to_model_action']('ACTION7'), 'ACTION7')
        self.assertIsNone(after['to_engine_action']('UNDO'))
        self.assertEqual(patched.replace('    "ACTION7": "ACTION7",\n', ''), source.decode())
        for name in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'MOUSE', 'RESET', '', None,
                     'bad', 'ACTION8', *[f'ACTION{i}' for i in range(1, 7)]]:
            with self.subTest(name=name):
                for function in ['to_engine_action', 'to_model_action']:
                    self.assertEqual(after[function](name), before[function](name))
        valid = ['ACTION1', 'ACTION6', 'ACTION7', 'ACTION7']
        self.assertEqual(before['to_model_actions'](valid), after['to_model_actions'](valid))

    def test_pinned_solver_dispatch_and_live_action_validity(self):
        path = ROOT / 'build/flash-action7-source-20260919/solver.py'
        if not path.is_file():
            self.skipTest('download pinned solver.py as documented in the ACTION7 handoff')
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '65ddfcdbe8c24fdc31246ff018b51469827b21c38995243a36a818adbed33370')
        cls = next(n for n in ast.parse(raw).body
                   if isinstance(n, ast.ClassDef) and n.name == '_HarnessGameSession')
        names = {'_normalize_actions', 'step_env', '_error_payload'}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {}
        exec(compile('from __future__ import annotations\n' +
                     '\n'.join(ast.unparse(n) for n in methods), 'pinned-dispatch', 'exec'),
             namespace)
        stub = SimpleNamespace
        namespace.update(
            arcengine=stub(GameAction=stub(
                ACTION6=stub(name='ACTION6', value=6),
                from_name=lambda name: stub(name=name, value=int(name[6:]))),
                ActionInput=lambda id, data: stub(id=id, data=data)),
            _is_engine_game_over=lambda game: False,
            _engine_action_names=lambda game: [f'ACTION{i}' for i in game.current_state.available_actions],
            _format_action_display=lambda name, data: name,
        )
        for patched in [False, True]:
            with self.subTest(patched=patched):
                mapping = {}
                source = FIXTURE.read_bytes()
                exec(PATCH.patch_action_names(source) if patched else source, mapping)
                namespace.update({key: mapping[key] for key in ['to_engine_action', 'to_model_actions']})
                executed = []
                session = stub(game=stub(current_state=stub(available_actions=[7])),
                               should_stop=lambda: False, timing_payload=lambda: {},
                               write_viewer_payload=lambda: None)
                for name in names:
                    setattr(session, name, MethodType(namespace[name], session))

                def execute(action, **kwargs):
                    executed.append(action.id.name)
                    return {'executed': True, 'action_name': action.id.name, 'board_changed': True}

                session._execute_action = execute
                result = session.step_env({'actions': [{'action': 'ACTION7'}]})
                self.assertEqual(executed, ['ACTION7'] if patched else [])
                self.assertEqual(result['executed'], patched)
                # Parsing support never bypasses the engine's available-actions gate.
                session.game.current_state.available_actions = [1]
                result = session.step_env({'actions': [{'action': 'ACTION7'}]})
                self.assertFalse(result['executed'])
                self.assertEqual(executed, ['ACTION7'] if patched else [])

    def test_source_drift_and_double_application_fail_closed(self):
        source = FIXTURE.read_bytes()
        for changed in [source + b'\n', PATCH.patch_action_names(source).encode()]:
            with self.subTest(changed=hashlib.sha256(changed).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'digest changed'):
                    PATCH.patch_action_names(changed)

    def test_opt_in_and_single_variable_experiment(self):
        for games in [None, 25]:
            baseline = generated_notebook(max_games=games)
            candidate = generated_notebook(max_games=games, include_action7_patch=True)
            self.assertNotIn('FLASH_ACTION7', code_source(baseline))
            source = code_source(candidate)
            self.assertNotIn('FLASH_AGENT_STATE_PATCH', source)
            self.assertIn('Not a verified score improvement.', ''.join(candidate['cells'][0]['source']))
            # Only the setup cell changes; runtime, serving and gameplay remain baseline.
            for index, (before, after) in enumerate(zip(baseline['cells'], candidate['cells'])):
                if before['cell_type'] == 'code' and index != 9:
                    self.assertEqual(before['source'], after['source'])
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            generated_notebook(max_games=None, include_action7_patch=True,
                               include_agent_state_patch=True)

    def test_generated_staging_imports_patch_and_preserves_package_exports(self):
        # Run the actual generated setup staging/activation in a fresh interpreter.
        # Model setup is omitted; tiny modules stand in for heavyweight dependencies.
        nb = generated_notebook(max_games=None, include_action7_patch=True)
        setup = ''.join(nb['cells'][9]['source'])
        # Extract the patch module separately (no teardown fixture required).
        patch_source = (ROOT / 'scripts/flash_action7_patch.py').read_text()
        stage = setup[setup.index('_action7_source_path ='):setup.index('# Each bundled repo')]
        ready_start = setup.index('if any(name == "inference"')
        ready = setup[ready_start:setup.index('# Solver setup commands')]
        finish = setup[setup.index('# Honour any PYTHONPATH'):]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / 'bundle'
            base = bundle / 'src/ARC3-Inference'
            agent = base / 'inference/agent'
            agent.mkdir(parents=True)
            (base / 'inference/__init__.py').write_text('')
            (agent / '__init__.py').write_text('')
            (agent / 'action_names.py').write_bytes(FIXTURE.read_bytes())
            (agent / 'tool_agent.py').write_text(
                'from inference.agent.action_names import to_engine_action\n'
                'class ToolAgent:\n    action_name = staticmethod(to_engine_action)\n')
            (agent / 'runtime_state.py').write_text('class Frame: pass\nclass HistoryEntry: pass\n')
            work = root / 'working'
            work.mkdir()
            code = (
                'import os, sys, json, hashlib\nfrom pathlib import Path\n'
                f'BUNDLE_DIR = Path({str(bundle)!r})\nWORKING_DIR = Path({str(work)!r})\n'
                f'sys.path.insert(0, {str(base)!r})\n'
                + patch_source + '\n' + stage + '\n' + ready
                # Simulate setup putting the original source first again.
                + f'\nsys.path.insert(0, {str(base)!r})\n' + finish
                + '\nfrom inference.agent import ToolAgent, Frame, HistoryEntry\n'
                + 'assert ToolAgent.action_name("ACTION7") == "ACTION7"\n'
                + 'assert _action7_module.to_engine_action("ACTION8") is None\n'
            )
            result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                    text=True, check=True)
            self.assertIn('FLASH_ACTION7_OVERLAY_IMPORTED', result.stdout)
            record = json.loads((work / 'flash_action7_patch.json').read_text())
            self.assertEqual(record['patched_sha256'], AUDIT.ACTION7_PATCH_HASH)
            self.assertEqual((agent / 'action_names.py').read_bytes(), FIXTURE.read_bytes())
            # Import-order failure must be explicit, not a silently unused overlay.
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run([sys.executable, '-c', code.replace(
                    ready, 'import inference\n' + ready, 1)],
                    capture_output=True, text=True, check=True)


class Action7DiagnosticTests(unittest.TestCase):
    def test_count_only_tool_sections_not_metadata_and_report_missing_traces(self):
        diagnostic = load_script('diagnose_flash_action7')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'transcripts').mkdir()
            (root / 'benchmark.json').write_text(json.dumps({'game_runs': [
                {'game_id': 'one', 'history': [{'action': {'id': 'ACTION7'}}]},
                {'game_id': 'two', 'history': []},
            ]}))
            (root / 'transcripts/one_p0.txt').write_text(
                '\n[MODEL RESPONSE META]\nACTION7\n[TOOL CALL: python]\n'
                "action(['ACTION7', 'ACTION7'])\n[TOOL RESULT: python]\nACTION7\n")
            report = diagnostic.diagnose(root)
            self.assertEqual(report['tool_snippets_mentioning_action7'], 1)
            self.assertEqual(report['executed_action7'], 1)
            self.assertEqual(report['transcript_game_count'], 1)
            self.assertFalse(report['games'][1]['transcripts_present'])


if __name__ == '__main__':
    unittest.main()
