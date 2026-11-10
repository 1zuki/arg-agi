"""Digest-bound direct-tool retry patch for the pulled 3.40 Flash agent."""

import hashlib


PATCH_NAME = "direct-tool-retry-v1"
SOURCE_SHA256 = "535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d"


def patch_tool_agent(source: bytes) -> str:
    """Make a tool-less model response retry with a direct Python tool call."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Flash tool-agent source digest changed; refusing to patch.")

    text = source.decode("utf-8")
    old = '''                    followup_prefix = "You have not acted yet. Investigate first. "
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
                            "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
                            "Emit exactly one `python` tool call directly as your next response. "
                            "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                        )
'''
    new = '''                    followup_prefix = (
                        "You did not call a tool. On this retry, emit exactly one `python` tool call directly "
                        "as your next response. Do not add a note or explanation first. "
                    )
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
                            "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
                            "Emit exactly one `python` tool call directly as your next response. "
                            "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                        )
'''
    if text.count(old) != 1:
        raise ValueError("Flash direct-tool retry patch anchor did not match exactly once.")
    text = text.replace(old, new, 1)
    compile(text, "flash_tool_agent_direct_retry.py", "exec")
    return text
