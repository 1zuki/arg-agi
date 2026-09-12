#!/usr/bin/env python3
"""Read-only artifact audit for Flash-Next/MTP offline runs (not hidden scores)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PUBLIC_IDS = (
    'tn36-ef4dde99', 'lf52-271a04aa', 'cn04-2fe56bfb', 'bp35-0a0ad940',
    'wa30-ee6fef47', 'lp85-305b61c3', 'r11l-495a7899', 'tu93-0768757b',
    'sp80-589a99af', 'm0r0-492f87ba', 'vc33-5430563c', 'ar25-0c556536',
    'ka59-38d34dbb', 'sc25-635fd71a', 'sk48-d8078629', 'dc22-fdcac232',
    'cd82-fb555c5d', 'ft09-0d8bbf25', 'g50t-5849a774', 'ls20-9607627b',
    're86-8af5384d', 's5i5-18d95033', 'sb26-7fbdac44', 'su15-1944f8ab',
    'tr87-cd924810',
)
SOURCE_HASH = 'c48368e330abf2574b155b42041bd5d42ea556d53340ccbbc5c8788acaa0eb19'
PATCH_HASH = '5224fe6f6a3f8d4def01954a4a0de355a6746bc4cc289c082a0e07ac0efef725'
SERVED_MODEL = 'Qwen/Qwen3.8-Flash-Next-NVFP4'
TUNING = {
    'enable_chunked_prefill': True, 'enable_prefix_caching': False,
    'kv_cache_dtype': 'auto', 'kv_cache_memory_bytes': 5368709120,
    'max_cudagraph_capture_size': 32, 'max_num_batched_tokens': 8192,
    'max_num_seqs': 8, 'mtp_speculative_tokens': 3, 'omp_num_threads': 1,
}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def read(root: Path, name: str) -> bytes:
    path = root / name
    require(not path.is_symlink() and path.is_file(), f'{name}: missing or symlink')
    require(0 < path.stat().st_size <= 128 * 1024 * 1024, f'{name}: invalid size')
    return path.read_bytes()


def document(root: Path, name: str) -> dict:
    result = json.loads(read(root, name))
    require(isinstance(result, dict), f'{name}: expected JSON object')
    return result


def digest(root: Path, name: str) -> str:
    return hashlib.sha256(read(root, name)).hexdigest()


def audit(root: Path, mode: str, *, check_parquet: bool = True) -> dict:
    require(mode in ('preflight', 'full-offline'), 'unsupported audit mode')
    expected_ids = list(PUBLIC_IDS[:1] if mode == 'preflight' else PUBLIC_IDS)
    td = document(root, 'vllm-server-teardown.json')
    for key in ('shutdown_ok', 'identity_valid', 'working_dir_validated',
                'port_closed', 'final_metrics_preserved', 'required_artifacts_preserved'):
        require(td.get(key) is True, f'teardown.{key}')
    require(td.get('gpu_query_error_after') is False, 'teardown.gpu_query_error_after')
    for key in ('identity_errors', 'vllm_gpu_rows_after', 'full_proc_marker_survivors',
                'cpu_only_vllm_ple_marker_survivors'):
        require(td.get(key) == [], f'teardown.{key}')
    scan = td['process_scan_final_gate']
    for key in ('authorized_records', 'suspect_records', 'saved_conflicts'):
        require(scan.get(key) == [], f'teardown.scan.{key}')
    require('root_conflict' in scan and scan['root_conflict'] is None,
            'teardown.scan.root_conflict')
    for name, key in [('vllm-metrics-final.prom', 'final_metrics_sha256'),
                      ('vllm-models-final.json', 'final_models_sha256')]:
        require(digest(root, name) == td.get(key), f'{name}: digest mismatch')
    for name in ('score.json', 'submission.parquet'):
        before = td['required_artifacts_before'][name]
        require(before == td['required_artifacts_after'][name], f'{name}: changed at teardown')
        require(before.get('exists') is True and before.get('is_file') is True,
                f'{name}: absent at teardown')
        require(before.get('bytes') == len(read(root, name)) and
                before.get('sha256') == digest(root, name), f'{name}: downloaded digest mismatch')
    patch = document(root, 'flash_teardown_patch.json')
    require(patch == {'patch': 'gpu-release-settle-v1', 'source_sha256': SOURCE_HASH,
                      'patched_sha256': PATCH_HASH}, 'teardown patch identity mismatch')
    require(digest(root, 'flash_serving_teardown.py') == PATCH_HASH, 'patched source changed')
    provenance = document(root, 'vllm-setup-provenance.json')
    expected = {
        'model_hf_repo': 'RadixArk/Qwen3.8-Flash-Next-NVFP4',
        'model_hf_revision': '7b719225242aacd3dbd3f9407468c2ee9a9d2594',
        'vllm_version': '0.1.dev20073+g8e685d198',
        'source_identity_sha256': '473e695998342160478e9066d8a0d45536942ef37c75ddb803f36d3e0abb397c',
    }
    for key, value in expected.items():
        require(provenance.get(key) == value, f'provenance.{key}')
    require(provenance['model'].get('config_sha256') ==
            'e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624', 'model config identity')
    require(provenance['runtime'].get('manifest_sha256') ==
            'e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad', 'runtime identity')
    identity = document(root, 'vllm-server-identity.json')
    for key, value in TUNING.items():
        for config in (provenance['vllm_tuning'], identity['vllm_tuning']):
            require(config.get(key) == value, f'serving tuning.{key}')
    models = document(root, 'vllm-models-final.json')
    require([m['id'] for m in models['data']] == [SERVED_MODEL], 'served model mismatch')
    watchdog = document(root, 'vllm-watchdog-status.json')
    require(watchdog.get('event') == 'watchdog_stopped', 'watchdog did not stop')
    require(type(watchdog.get('restart_attempts')) is int and
            0 <= watchdog['restart_attempts'] <= 2, 'invalid watchdog restart count')

    bm = document(root, 'benchmark.json')
    require(bm.get('n_passes') == 1 and isinstance(bm.get('end_time'), str), 'benchmark incomplete')
    runs = bm['game_runs']
    ids = [r['game_id'] for r in runs]
    require(ids == expected_ids, 'incorrect game coverage or order')
    states = {'won', 'gave_up', 'cancelled'} if mode == 'preflight' else {'won', 'gave_up'}
    for run in runs:
        require(run.get('state') in states, 'crashed, cancelled, or unfinished run')
        require(finite(run.get('final_score')) and run['final_score'] >= 0, 'invalid final score')
        require(isinstance(run.get('history'), list) and len(run['history']) > 0, 'run has no actions')
        require(finite(run.get('final_wallclock_seconds')) and run['final_wallclock_seconds'] > 0,
                'invalid gameplay wallclock')
    score = document(root, 'score.json')
    require(score['metadata']['scoring_version'] == 'taaf-framework-score-v1', 'wrong scorer')
    require(set(score['metadata']['game_ids']) == set(ids) and
            score['metadata']['game_count'] == len(ids) and set(score['games']) == set(ids), 'scorer coverage')
    for run in runs:
        value = score['games'][run['game_id']]['score']
        require(finite(value) and math.isclose(value, run['final_score'], abs_tol=1e-8), 'per-game score mismatch')
    mean = sum(r['final_score'] for r in runs) / len(runs)
    require(finite(score['score']) and math.isclose(score['score'], mean, abs_tol=1e-8), 'mean score mismatch')
    if check_parquet:
        import pyarrow.parquet as pq
        table = pq.read_table(root / 'submission.parquet')
        require(table.column_names == ['row_id', 'game_id', 'end_of_game', 'score'], 'parquet columns')
        require(table.to_pylist() == [{'row_id': '1_0', 'game_id': '1', 'end_of_game': True, 'score': 1}],
                'incorrect offline placeholder')
    log = json.loads(read(root, 'arc-agi3-flash-next-mtp-' + ('preflight' if mode == 'preflight' else 'full') + '.log'))
    text = ''.join(row.get('data', '') for row in log)
    budget = 1800 if mode == 'preflight' else 7920
    require(f'PUBLIC25_SETTINGS budget_s={budget}.0 concurrency=28 analyzer_timeout=900.0' in text,
            'runtime configuration mismatch')
    require(f'PUBLIC25_AUDIT runs={len(ids)} actions=' in text, 'offline audit log missing')
    require('Traceback (most recent call last)' not in text, 'Python traceback in kernel log')
    warnings = ['Offline placeholder is not a scored submission; kernel completion must be checked separately.']
    if not provenance['model'].get('payload_sha256_verified') or not provenance['runtime'].get('payload_sha256_verified'):
        warnings.append('Fast setup checks identity, not all model/runtime payload hashes.')
    if any(r['state'] == 'cancelled' for r in runs):
        warnings.append('Preflight deadline cancellation accepted only for the runtime smoke test.')
    return {'passed': True, 'mode': mode, 'game_count': len(runs), 'offline_mean': mean,
            'total_actions': sum(len(r['history']) for r in runs),
            'watchdog_restarts': watchdog['restart_attempts'], 'parquet_checked': check_parquet,
            'shutdown_ok': True, 'warnings': warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--mode', choices=('preflight', 'full-offline'), required=True)
    args = parser.parse_args()
    try:
        report = audit(args.output_dir, args.mode)
    except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError) as error:
        # Avoid echoing logs or arbitrary data from downloaded artifacts.
        print(json.dumps({'passed': False, 'error_type': type(error).__name__}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
