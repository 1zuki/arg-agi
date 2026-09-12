from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audit_flash_output', ROOT / 'scripts/audit_flash_output.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FlashAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def change(self, name, edit):
        path = self.root / name
        doc = json.loads(path.read_text())
        edit(doc)
        self.write(name, doc)

    def fixture(self, full=False):
        ids = list(AUDIT.PUBLIC_IDS if full else AUDIT.PUBLIC_IDS[:1])
        self.write('score.json', {
            'score': 1.0, 'games': {g: {'score': 1.0} for g in ids},
            'metadata': {'scoring_version': 'taaf-framework-score-v1', 'game_ids': ids,
                         'game_count': len(ids)},
        })
        (self.root / 'submission.parquet').write_bytes(b'PAR1fixturePAR1')
        (self.root / 'vllm-metrics-final.prom').write_text('mock_metric 1\n')
        self.write('vllm-models-final.json', {'data': [{'id': AUDIT.SERVED_MODEL}]})
        # No executable file is needed: patch identity is checked with a mocked
        # digest for this one file, while artifact/metrics hashes remain real.
        self.write('flash_teardown_patch.json', {
            'patch': 'gpu-release-settle-v1', 'source_sha256': AUDIT.SOURCE_HASH,
            'patched_sha256': AUDIT.PATCH_HASH,
        })
        from unittest.mock import patch
        original_digest = AUDIT.digest
        mock = patch.object(AUDIT, 'digest', side_effect=lambda root, name:
                            AUDIT.PATCH_HASH if name == 'flash_serving_teardown.py'
                            else original_digest(root, name))
        mock.start()
        self.addCleanup(mock.stop)
        artifacts = {n: {'exists': True, 'is_file': True,
                         'bytes': (self.root / n).stat().st_size,
                         'sha256': original_digest(self.root, n)}
                     for n in ('score.json', 'submission.parquet')}
        td = {k: True for k in ('shutdown_ok', 'identity_valid', 'working_dir_validated',
                               'port_closed', 'final_metrics_preserved', 'required_artifacts_preserved')}
        td.update({k: [] for k in ('identity_errors', 'vllm_gpu_rows_after',
                                  'full_proc_marker_survivors', 'cpu_only_vllm_ple_marker_survivors')})
        td.update(gpu_query_error_after=False, process_scan_final_gate={
            'authorized_records': [], 'suspect_records': [], 'saved_conflicts': [], 'root_conflict': None,
        }, required_artifacts_before=artifacts, required_artifacts_after=artifacts,
            final_metrics_sha256=original_digest(self.root, 'vllm-metrics-final.prom'),
            final_models_sha256=original_digest(self.root, 'vllm-models-final.json'))
        self.write('vllm-server-teardown.json', td)
        self.write('vllm-setup-provenance.json', {
            'model_hf_repo': 'RadixArk/Qwen3.8-Flash-Next-NVFP4',
            'model_hf_revision': '7b719225242aacd3dbd3f9407468c2ee9a9d2594',
            'vllm_version': '0.1.dev20073+g8e685d198',
            'source_identity_sha256': '473e695998342160478e9066d8a0d45536942ef37c75ddb803f36d3e0abb397c',
            'model': {'config_sha256': 'e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624'},
            'runtime': {'manifest_sha256': 'e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad'},
            'vllm_tuning': AUDIT.TUNING,
        })
        self.write('vllm-server-identity.json', {'vllm_tuning': AUDIT.TUNING})
        self.write('vllm-watchdog-status.json', {'event': 'watchdog_stopped', 'restart_attempts': 0})
        self.write('benchmark.json', {'n_passes': 1, 'end_time': '2026-09-11T00:00:00',
            'game_runs': [{'game_id': g, 'state': 'gave_up', 'final_score': 1.0,
                           'history': [{}], 'final_wallclock_seconds': 50.0} for g in ids]})
        suffix = 'full' if full else 'preflight'
        self.write(f'arc-agi3-flash-next-mtp-{suffix}.log', [{'data':
            f'PUBLIC25_SETTINGS budget_s={7920 if full else 1800}.0 concurrency=28 analyzer_timeout=900.0\n'
            f'PUBLIC25_AUDIT runs={len(ids)} actions=25\n'}])

    def run_audit(self, full=False):
        return AUDIT.audit(self.root, 'full-offline' if full else 'preflight', check_parquet=False)

    def test_valid_preflight(self):
        self.fixture()
        self.assertTrue(self.run_audit()['passed'])

    def test_valid_full(self):
        self.fixture(full=True)
        self.assertEqual(self.run_audit(full=True)['game_count'], 25)

    def test_full_rejects_cancelled_game(self):
        self.fixture(full=True)
        self.change('benchmark.json', lambda d: d['game_runs'][0].update(state='cancelled'))
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            self.run_audit(full=True)

    def test_preflight_accepts_deadline_cancellation(self):
        self.fixture()
        self.change('benchmark.json', lambda d: d['game_runs'][0].update(state='cancelled'))
        self.assertTrue(self.run_audit()['passed'])

    def test_metrics_tamper_rejected(self):
        self.fixture()
        (self.root / 'vllm-metrics-final.prom').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.run_audit()

    def test_gpu_query_failure_rejected(self):
        self.fixture()
        self.change('vllm-server-teardown.json', lambda d: d.update(gpu_query_error_after=True))
        with self.assertRaisesRegex(ValueError, 'gpu_query_error_after'):
            self.run_audit()

    def test_missing_game_coverage_rejected(self):
        self.fixture(full=True)
        self.change('benchmark.json', lambda d: d['game_runs'].pop())
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.run_audit(full=True)

    def test_wrong_model_rejected(self):
        self.fixture()
        self.change('vllm-setup-provenance.json', lambda d: d.update(model_hf_revision='wrong'))
        with self.assertRaisesRegex(ValueError, 'model_hf_revision'):
            self.run_audit()

    def test_nan_score_rejected(self):
        self.fixture()
        self.change('benchmark.json', lambda d: d['game_runs'][0].update(final_score=float('nan')))
        with self.assertRaisesRegex(ValueError, 'invalid final score'):
            self.run_audit()

    def test_crashed_game_rejected_in_preflight(self):
        self.fixture()
        self.change('benchmark.json', lambda d: d['game_runs'][0].update(state='crashed'))
        with self.assertRaisesRegex(ValueError, 'crashed'):
            self.run_audit()

    def test_duplicate_game_ids_rejected(self):
        self.fixture(full=True)
        self.change('benchmark.json', lambda d: d['game_runs'][1].update(
            game_id=d['game_runs'][0]['game_id']))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.run_audit(full=True)

    def test_submission_tamper_rejected(self):
        self.fixture()
        (self.root / 'submission.parquet').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'downloaded digest mismatch'):
            self.run_audit()

    def test_symlink_rejected(self):
        self.fixture()
        metrics = self.root / 'vllm-metrics-final.prom'
        metrics.rename(self.root / 'metrics.original')
        metrics.symlink_to(self.root / 'metrics.original')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_audit()
