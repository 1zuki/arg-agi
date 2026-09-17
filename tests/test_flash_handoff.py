from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'handoff_builder', ROOT / 'scripts/build_flash_next_package.py'
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
HANDOFF = ROOT / 'kaggle/flash-next-nostate-c12'


def code_cells(notebook):
    return [''.join(cell.get('source', [])) for cell in notebook['cells']
            if cell.get('cell_type') == 'code']


class FlashHandoffTests(unittest.TestCase):
    def test_published_packages_match_builder_code_and_safe_metadata(self):
        for mode in ('preflight', 'full'):
            with self.subTest(mode=mode):
                folder = HANDOFF / mode
                meta = json.loads((folder / 'kernel-metadata.json').read_text())
                published = json.loads((folder / meta['code_file']).read_text())
                generated = json.loads(BUILDER.SOURCE_NOTEBOOK.read_text())
                settings = dict(max_games=None, concurrency=12)
                if mode == 'preflight':
                    settings.update(max_games=25, runtime_seconds=1800,
                                    terminal_grace_seconds=120, analyzer_timeout=900)
                BUILDER._add_preflight_support(generated, **settings)
                self.assertEqual(code_cells(published), code_cells(generated))
                BUILDER._compile_notebook(published, f'published-{mode}')
                for cell in published['cells']:
                    if cell.get('cell_type') == 'code':
                        self.assertIsNone(cell['execution_count'])
                        self.assertEqual(cell['outputs'], [])
                source = '\n'.join(code_cells(published))
                self.assertNotIn('FLASH_AGENT_STATE_PATCH', source)
                self.assertIn('Not a verified score improvement.', ''.join(published['cells'][0]['source']))
                self.assertTrue(meta['is_private'])
                self.assertTrue(meta['enable_gpu'])
                self.assertFalse(meta['enable_internet'])
                self.assertEqual(meta['machine_shape'], BUILDER.MACHINE_SHAPE)
                self.assertEqual(meta['dataset_sources'], BUILDER.DATASET_SOURCES)
                self.assertEqual(meta['model_sources'], BUILDER.MODEL_SOURCES)
                self.assertEqual(meta['competition_sources'], [BUILDER.COMPETITION_SOURCE])
                self.assertEqual(meta['docker_image'], BUILDER.DOCKER_IMAGE)
                self.assertTrue(meta['id'].startswith('your-kaggle-username/'))
                self.assertEqual(BUILDER._slugify(meta['title']), meta['id'].split('/')[1])

    def test_full_handoff_has_uncapped_offline_settings_and_gateway_deadline(self):
        nb = json.loads((HANDOFF / 'full/arc-agi3-qwen38-flash-next-mtp-full.ipynb').read_text())
        source = '\n'.join(code_cells(nb))
        self.assertIn('bm.solver.max_runtime_s_per_game = 7920.0', source)
        self.assertIn('bm.solver.analyzer_timeout = 900.0', source)
        self.assertIn('FLASH_RUNTIME_CONCURRENCY", "12"', source)
        self.assertNotIn('PUBLIC25_DEADLINE origin=post_setup', source)
        assignments = [node for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == 'soft_end'
                           for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value.test), 'TRUE_SUBMISSION')
        self.assertIsNone(assignments[0].value.body.value)
