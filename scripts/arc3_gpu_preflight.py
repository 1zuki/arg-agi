"""Report whether the local machine can plausibly host the ARC-AGI-3 profile.

This is an advisory, side-effect-free check.  It does not import a model
runtime, allocate CUDA memory, download weights, start a server, or contact
Kaggle.  A passing result only means that the declared hardware thresholds
are met; it is not a gameplay or score result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuRecord:
    name: str
    total_mib: int
    free_mib: int | None


def _nvidia_smi() -> tuple[list[GpuRecord], str | None]:
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found"
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi failed: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return [], f"nvidia-smi failed: {detail}"
    records: list[GpuRecord] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            return [], f"unparseable nvidia-smi row: {line!r}"
        try:
            total = int(fields[1])
            free = int(fields[2])
        except ValueError:
            return [], f"non-numeric nvidia-smi row: {line!r}"
        records.append(GpuRecord(fields[0], total, free))
    return records, None if records else "nvidia-smi returned no GPUs"


def run_preflight(
    *,
    minimum_total_mib: int = 32_000,
    minimum_free_mib: int = 24_000,
) -> dict[str, Any]:
    """Return a JSON-safe local resource report.

    The thresholds are intentionally conservative planning gates, not a claim
    that any particular GPU is sufficient for every serving configuration.
    """

    if minimum_total_mib < 1 or minimum_free_mib < 1:
        raise ValueError("GPU memory thresholds must be positive")
    gpus, error = _nvidia_smi()
    eligible = [
        gpu
        for gpu in gpus
        if gpu.total_mib >= minimum_total_mib
        and (gpu.free_mib is None or gpu.free_mib >= minimum_free_mib)
    ]
    return {
        "check": "arc3_gpu_preflight",
        "side_effects": [],
        "thresholds_mib": {
            "minimum_total": minimum_total_mib,
            "minimum_free": minimum_free_mib,
        },
        "gpus": [asdict(gpu) for gpu in gpus],
        "eligible_gpus": [asdict(gpu) for gpu in eligible],
        "probe_error": error,
        "ready": bool(eligible),
        "interpretation": (
            "Hardware gate passed; run the exact one-game preflight before any full run."
            if eligible
            else "Hardware gate not met; do not start the model-serving run on this host."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-total-mib", type=int, default=32_000)
    parser.add_argument("--minimum-free-mib", type=int, default=24_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_preflight(
        minimum_total_mib=args.minimum_total_mib,
        minimum_free_mib=args.minimum_free_mib,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
