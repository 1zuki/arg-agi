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

    def fixture(self, full=False, game_count=None, concurrency=28, game_id=None,
                runtime_seconds=None, deadline_origin=None):
        count = len(AUDIT.PUBLIC_IDS) if full else (game_count or 1)
        ids = [game_id] if game_id is not None else list(AUDIT.PUBLIC_IDS[:count])
        runtime_seconds = runtime_seconds or (7920 if full else 1800)
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
        deadline = (
            f'PUBLIC25_DEADLINE origin={deadline_origin} '
            f'gameplay_budget_s={runtime_seconds}.0\n'
            if deadline_origin is not None else ''
        )
        self.write(f'arc-agi3-flash-next-mtp-{suffix}.log', [{'data':
            f'PUBLIC25_SETTINGS budget_s={runtime_seconds}.0 concurrency={concurrency} analyzer_timeout=900.0\n'
            f'{deadline}PUBLIC25_AUDIT runs={len(ids)} actions=25\n'}])

    def run_audit(self, full=False, expected_games=None, expected_concurrency=None,
                  expected_game_id=None, expected_runtime_seconds=None,
                  require_clean=False, require_runtime_from_ready=False):
        return AUDIT.audit(
            self.root,
            'full-offline' if full else 'preflight',
            check_parquet=False,
            expected_games=expected_games,
            expected_concurrency=expected_concurrency,
            expected_game_id=expected_game_id,
            expected_runtime_seconds=expected_runtime_seconds,
            require_clean=require_clean,
            require_runtime_from_ready=require_runtime_from_ready,
        )

    def test_valid_preflight(self):
        self.fixture()
        self.assertTrue(self.run_audit()['passed'])

    def test_valid_full(self):
        self.fixture(full=True)
        self.assertEqual(self.run_audit(full=True)['game_count'], 25)

    def test_custom_preflight_settings_are_checked(self):
        self.fixture(game_count=8, concurrency=8)
        (self.root / 'arc-agi3-flash-next-mtp-preflight.log').rename(
            self.root / 'arc-agi3-flash-next-mtp-preflight-c8.log'
        )
        report = self.run_audit(expected_games=8, expected_concurrency=8)
        self.assertEqual(report['game_count'], 8)
        self.assertEqual(report['expected_concurrency'], 8)
        self.assertEqual(report['kernel_log'], 'arc-agi3-flash-next-mtp-preflight-c8.log')

    def test_variant_before_mode_log_is_selected(self):
        self.fixture(full=True, concurrency=8)
        (self.root / 'arc-agi3-flash-next-mtp-full.log').rename(
            self.root / 'arc-agi3-flash-next-mtp-c8-full.log'
        )
        report = self.run_audit(full=True, expected_concurrency=8)
        self.assertEqual(report['kernel_log'], 'arc-agi3-flash-next-mtp-c8-full.log')

    def test_custom_preflight_concurrency_mismatch_rejected(self):
        self.fixture(game_count=8, concurrency=28)
        with self.assertRaisesRegex(ValueError, 'runtime configuration mismatch'):
            self.run_audit(expected_games=8, expected_concurrency=8)

    def test_isolated_nonfirst_game_requires_clean_post_setup_run(self):
        self.fixture(
            concurrency=8,
            game_id='tr87-cd924810',
            runtime_seconds=7920,
            deadline_origin='post_setup',
        )
        report = self.run_audit(
            expected_games=1,
            expected_game_id='tr87-cd924810',
            expected_concurrency=8,
            expected_runtime_seconds=7920,
            require_clean=True,
            require_runtime_from_ready=True,
        )
        self.assertEqual(report['expected_game_id'], 'tr87-cd924810')
        self.assertTrue(report['require_clean'])

    def test_clean_isolated_preflight_rejects_cancellation(self):
        self.fixture(
            concurrency=8,
            game_id='tr87-cd924810',
            runtime_seconds=7920,
            deadline_origin='post_setup',
        )
        self.change('benchmark.json', lambda d: d['game_runs'][0].update(state='cancelled'))
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            self.run_audit(
                expected_game_id='tr87-cd924810',
                expected_concurrency=8,
                expected_runtime_seconds=7920,
                require_clean=True,
                require_runtime_from_ready=True,
            )

    def test_ambiguous_kernel_logs_rejected(self):
        self.fixture()
        (self.root / 'arc-agi3-flash-next-mtp-preflight-copy.log').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.run_audit()

    def test_full_audit_accepts_custom_concurrency(self):
        self.fixture(full=True, concurrency=8)
        report = self.run_audit(full=True, expected_concurrency=8)
        self.assertEqual(report['expected_concurrency'], 8)

    def test_full_audit_rejects_custom_game_count(self):
        self.fixture(full=True)
        with self.assertRaisesRegex(ValueError, 'full-offline audit requires'):
            self.run_audit(full=True, expected_games=8, expected_concurrency=8)

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
