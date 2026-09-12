"""Narrow patch for Keith Tyser's pinned Flash-Next/MTP serving teardown."""


def patch_flash_teardown(source: bytes) -> str:
    """Wait for delayed GPU release without repeating capture or relaxing gates."""
    import hashlib

    expected = "c48368e330abf2574b155b42041bd5d42ea556d53340ccbbc5c8788acaa0eb19"
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Flash teardown source digest changed; refusing to patch.")
    text = source.decode("utf-8")
    old_scan = '''    if identity is not None:
        final_scan = scan_ownership(identity, owned_process_table(identity))
        result["process_scan_final_gate"] = {
            key: value for key, value in final_scan.items() if key != "authorized"
        }
    full_marker_scan = global_marker_records()
    gpu_after = gpu_rows()
'''
    new_scan = '''    # CUDA allocations can outlive the signalled process briefly. Capture
    # endpoints only once, then poll GPU release before the original final gate.
    gpu_wait_started = time.monotonic()
    gpu_deadline = gpu_wait_started + GPU_RELEASE_GRACE_SECONDS
    gpu_after = gpu_rows()
    gpu_poll_count = 1
    while (
        not any("query_error" in row for row in gpu_after)
        and classify_vllm_gpu_rows(gpu_after, historical)
    ):
        remaining = gpu_deadline - time.monotonic()
        if remaining <= GPU_QUERY_TIMEOUT_SECONDS:
            break
        time.sleep(min(PROCESS_POLL_SECONDS, remaining - GPU_QUERY_TIMEOUT_SECONDS))
        gpu_after = gpu_rows()
        gpu_poll_count += 1
    result["gpu_release_wait_seconds"] = time.monotonic() - gpu_wait_started
    result["gpu_release_wait_bound_seconds"] = GPU_RELEASE_GRACE_SECONDS
    result["gpu_release_poll_count"] = gpu_poll_count

    # Recheck ownership and the port after waiting; never trust stale snapshots.
    if identity is not None:
        final_scan = scan_ownership(identity, owned_process_table(identity))
        result["process_scan_final_gate"] = {
            key: value for key, value in final_scan.items() if key != "authorized"
        }
    result["port_closed"] = not port_open()
    full_marker_scan = global_marker_records()
'''
    replacements = (
        ("GPU_QUERY_TIMEOUT_SECONDS = 3.0\n",
         "GPU_QUERY_TIMEOUT_SECONDS = 3.0\nGPU_RELEASE_GRACE_SECONDS = 12.0\n"),
        ("    + GPU_QUERY_TIMEOUT_SECONDS\n",
         "    + GPU_QUERY_TIMEOUT_SECONDS\n    + GPU_RELEASE_GRACE_SECONDS\n"),
        (old_scan, new_scan),
        ("assert MAX_EXPLICIT_WAIT_SECONDS == 11.3",
         "assert MAX_EXPLICIT_WAIT_SECONDS == 23.3"),
        ("assert MAX_EXPLICIT_WAIT_SECONDS <= 20.0",
         "assert MAX_EXPLICIT_WAIT_SECONDS <= 30.0"),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Flash teardown patch anchor did not match exactly once.")
        text = text.replace(old, new, 1)
    compile(text, "flash_serving_teardown.py", "exec")
    return text
