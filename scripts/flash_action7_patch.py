"""Digest-bound ACTION7 passthrough patch for the pinned Flash Duck adapter."""

import hashlib


PATCH_NAME = "action7-passthrough-v1"
SOURCE_SHA256 = "d359c22dd9f3925ed54cb05a60640e8aa1121c226a068b4736e8dfad46d2d115"


def patch_action_names(source: bytes) -> str:
    """Expose the engine's ACTION7 under its existing model-facing name."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Flash action-name source digest changed; refusing to patch.")
    text = source.decode("utf-8")
    old = '    "ACTION6": "MOUSE",\n    "RESET": "RESET",\n'
    new = '    "ACTION6": "MOUSE",\n    "ACTION7": "ACTION7",\n    "RESET": "RESET",\n'
    if text.count(old) != 1:
        raise ValueError("Flash ACTION7 patch anchor did not match exactly once.")
    text = text.replace(old, new, 1)
    compile(text, "flash_action_names.py", "exec")
    return text
