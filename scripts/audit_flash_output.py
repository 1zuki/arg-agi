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
AGENT_PATCH_NAME = 'reasoning-world-model-v1'
AGENT_SOURCE_HASH = '535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d'
AGENT_PATCH_HASH = '978026a51c438744c64922571b2264f09a3d4ec4259ddc75b03ccc0f3ba0b772'
AGENT_OVERLAY_PATH = 'flash_agent_overlay/inference/agent/tool_agent.py'
MAX_TERMINAL_GRACE_SECONDS = 599
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


def log_json_records(text: str, marker: str) -> list[dict]:
    records: list[dict] = []
    prefix = f'{marker} '
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        record = json.loads(line[len(prefix):])
        require(isinstance(record, dict), f'{marker}: expected JSON object')
        records.append(record)
    return records


def audit(
    root: Path,
    mode: str,
    *,
    check_parquet: bool = True,
    expected_games: int | None = None,
    expected_concurrency: int | None = None,
    expected_game_id: str | None = None,
    expected_runtime_seconds: int | None = None,
    require_clean: bool = False,
    require_runtime_from_ready: bool = False,
    expected_terminal_grace_seconds: int | None = None,
    require_agent_state_patch: bool = False,
    expected_analyzer_timeout: int = 900,
    kernel_log: str | None = None,
    expected_gameplay_budget_seconds: int | None = None,
) -> dict:
    require(mode in ('preflight', 'full-offline'), 'unsupported audit mode')
    default_games = 1 if mode == 'preflight' else len(PUBLIC_IDS)
    expected_games = default_games if expected_games is None else expected_games
    expected_concurrency = 28 if expected_concurrency is None else expected_concurrency
    default_runtime_seconds = 1800 if mode == 'preflight' else 7920
    expected_runtime_seconds = (
        default_runtime_seconds if expected_runtime_seconds is None else expected_runtime_seconds
    )
    expected_gameplay_budget_seconds = (
        expected_runtime_seconds
        if expected_gameplay_budget_seconds is None
        else expected_gameplay_budget_seconds
    )
    require(type(expected_games) is int and 1 <= expected_games <= len(PUBLIC_IDS),
            'invalid expected game count')
    require(type(expected_concurrency) is int and expected_concurrency >= 1,
            'invalid expected concurrency')
    require(type(expected_runtime_seconds) is int and expected_runtime_seconds >= 1,
            'invalid expected runtime seconds')
    require(type(expected_gameplay_budget_seconds) is int and expected_gameplay_budget_seconds >= 1,
            'invalid expected gameplay budget seconds')
    require(type(expected_analyzer_timeout) is int and expected_analyzer_timeout >= 1,
            'invalid expected analyzer timeout')
    require(expected_terminal_grace_seconds is None or (
        mode == 'preflight'
        and type(expected_terminal_grace_seconds) is int
        and 0 <= expected_terminal_grace_seconds <= MAX_TERMINAL_GRACE_SECONDS
    ), 'invalid expected terminal grace seconds')
    require(mode == 'preflight' or expected_games == len(PUBLIC_IDS),
            'full-offline audit requires 25 games')
    require(expected_game_id is None or mode == 'preflight',
            'expected game ID is only valid for preflight')
    require(expected_game_id is None or expected_game_id in PUBLIC_IDS,
            'expected game ID is not pinned')
    require(expected_game_id is None or expected_games == 1,
            'expected game ID requires one expected game')
    require(not require_runtime_from_ready or expected_game_id is not None,
            'runtime-from-ready requires an expected game ID')
    expected_ids = (
        [expected_game_id] if expected_game_id is not None
        else list(PUBLIC_IDS[:expected_games] if mode == 'preflight' else PUBLIC_IDS)
    )
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
    expected_agent_patch = {
        'patch': AGENT_PATCH_NAME,
        'source_sha256': AGENT_SOURCE_HASH,
        'patched_sha256': AGENT_PATCH_HASH,
        'overlay_tool_agent_sha256': AGENT_PATCH_HASH,
    }
    if require_agent_state_patch:
        require(document(root, 'flash_agent_state_patch.json') == expected_agent_patch,
                'agent state patch identity mismatch')
        require(digest(root, AGENT_OVERLAY_PATH) == AGENT_PATCH_HASH,
                'agent state overlay source changed')
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
    states = {'won', 'gave_up'} if require_clean or mode == 'full-offline' else {
        'won', 'gave_up', 'cancelled'
    }
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
    log_mode = 'preflight' if mode == 'preflight' else 'full'
    log_prefix = 'arc-agi3-flash-next-mtp-'
    log_names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.name.startswith(log_prefix)
        and path.suffix == '.log'
        and log_mode in path.stem[len(log_prefix):].split('-')
    )
    if kernel_log is not None:
        require(isinstance(kernel_log, str) and Path(kernel_log).name == kernel_log
                and kernel_log.startswith(log_prefix) and kernel_log.endswith('.log'),
                'invalid explicit kernel log name')
        log_names = [kernel_log]
    require(len(log_names) == 1, 'expected exactly one mode-matching kernel log')
    log = json.loads(read(root, log_names[0]))
    text = ''.join(row.get('data', '') for row in log)
    require(f'PUBLIC25_SETTINGS budget_s={expected_runtime_seconds}.0 concurrency={expected_concurrency} '
            f'analyzer_timeout={float(expected_analyzer_timeout)}' in text,
            'runtime configuration mismatch')
    if require_runtime_from_ready:
        require(
            f'PUBLIC25_DEADLINE origin=post_setup gameplay_budget_s={expected_gameplay_budget_seconds}.0' in text,
            'runtime deadline did not start after setup',
        )
    if expected_terminal_grace_seconds is not None:
        require(
            f'PUBLIC25_DEADLINE origin=post_setup gameplay_budget_s={expected_gameplay_budget_seconds}.0 '
            f'terminal_grace_s={float(expected_terminal_grace_seconds)}' in text,
            'terminal grace configuration mismatch',
        )
    if require_agent_state_patch:
        require(expected_agent_patch in log_json_records(text, 'FLASH_AGENT_STATE_PATCH'),
                'agent state patch log missing or mismatched')
        overlay_records = log_json_records(text, 'FLASH_AGENT_OVERLAY_READY')
        require(any(
            isinstance(record.get('root'), str)
            and record.get('tool_agent_sha256') == AGENT_PATCH_HASH
            for record in overlay_records
        ), 'agent state overlay log missing or mismatched')
        require('FLASH_AGENT_OVERLAY_REASSERTED' in text,
                'agent state overlay was not reasserted after setup')
        imported_records = log_json_records(text, 'FLASH_AGENT_OVERLAY_IMPORTED')
        require(any(
            isinstance(record.get('tool_agent_path'), str)
            and record['tool_agent_path'].endswith(AGENT_OVERLAY_PATH)
            and record.get('tool_agent_sha256') == AGENT_PATCH_HASH
            for record in imported_records
        ), 'agent state overlay import log missing or mismatched')
    require(f'PUBLIC25_AUDIT runs={len(ids)} actions=' in text, 'offline audit log missing')
    require('Traceback (most recent call last)' not in text, 'Python traceback in kernel log')
    warnings = ['Offline placeholder is not a scored submission; kernel completion must be checked separately.']
    if not provenance['model'].get('payload_sha256_verified') or not provenance['runtime'].get('payload_sha256_verified'):
        warnings.append('Fast setup checks identity, not all model/runtime payload hashes.')
    if any(r['state'] == 'cancelled' for r in runs):
        warnings.append('Preflight deadline cancellation accepted only for the runtime smoke test.')
    return {'passed': True, 'mode': mode, 'kernel_log': log_names[0], 'game_count': len(runs),
            'expected_game_id': expected_game_id,
            'expected_concurrency': expected_concurrency,
            'expected_runtime_seconds': expected_runtime_seconds,
            'expected_gameplay_budget_seconds': expected_gameplay_budget_seconds,
            'expected_analyzer_timeout': expected_analyzer_timeout,
            'expected_terminal_grace_seconds': expected_terminal_grace_seconds,
            'agent_state_patch_required': require_agent_state_patch,
            'require_clean': require_clean,
            'offline_mean': mean,
            'total_actions': sum(len(r['history']) for r in runs),
            'watchdog_restarts': watchdog['restart_attempts'], 'parquet_checked': check_parquet,
            'shutdown_ok': True, 'warnings': warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--mode', choices=('preflight', 'full-offline'), required=True)
    parser.add_argument('--expected-games', type=int,
                        help='Expected preflight game count (default: 1).')
    parser.add_argument('--expected-concurrency', type=int,
                        help='Expected preflight Duck concurrency (default: 28).')
    parser.add_argument('--expected-game-id',
                        help='Pinned single-game preflight ID to require.')
    parser.add_argument('--expected-runtime-seconds', type=int,
                        help='Expected per-game runtime (default: 1800 preflight, 7920 full).')
    parser.add_argument('--expected-gameplay-budget-seconds', type=int,
                        help='Expected total preflight gameplay budget after scheduling batches.')
    parser.add_argument('--require-clean', action='store_true',
                        help='Reject cancelled games, including in preflight mode.')
    parser.add_argument('--require-runtime-from-ready', action='store_true',
                        help='Require an isolated preflight deadline to start after setup.')
    parser.add_argument('--expected-terminal-grace-seconds', type=int,
                        help='Require this bounded post-runtime grace in a preflight log.')
    parser.add_argument('--expected-analyzer-timeout', type=int, default=900,
                        help='Expected analyzer request timeout (default: 900).')
    parser.add_argument('--kernel-log',
                        help='Exact downloaded kernel log filename when its slug omits the mode.')
    parser.add_argument('--require-agent-state-patch', action='store_true',
                        help='Require the digest-bound Flash reasoning-state overlay artifacts and logs.')
    args = parser.parse_args()
    try:
        report = audit(args.output_dir, args.mode,
                       expected_games=args.expected_games,
                       expected_concurrency=args.expected_concurrency,
                       expected_game_id=args.expected_game_id,
                       expected_runtime_seconds=args.expected_runtime_seconds,
                       expected_gameplay_budget_seconds=args.expected_gameplay_budget_seconds,
                       require_clean=args.require_clean,
                       require_runtime_from_ready=args.require_runtime_from_ready,
                       expected_terminal_grace_seconds=args.expected_terminal_grace_seconds,
                       require_agent_state_patch=args.require_agent_state_patch,
                       expected_analyzer_timeout=args.expected_analyzer_timeout,
                       kernel_log=args.kernel_log)
    except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError) as error:
        # Avoid echoing logs or arbitrary data from downloaded artifacts.
        print(json.dumps({'passed': False, 'error_type': type(error).__name__}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
