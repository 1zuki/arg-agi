"""Digest-bound state-persistence patch for the pinned Flash Duck agent."""

import hashlib


PATCH_NAME = "reasoning-world-model-v1"
SOURCE_SHA256 = "535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d"


def patch_tool_agent(source: bytes) -> str:
    """Preserve labelled state emitted in a tool-calling model's reasoning field."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Flash tool-agent source digest changed; refusing to patch.")

    text = source.decode("utf-8")
    label_parser_old = '''        lowered = candidate.lower()

        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break
'''
    label_parser_new = '''        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            label = target[:-1]
            match = re.match(
                rf"^{re.escape(label)}(?:\\s*\\([^:\\n]*\\))?\\s*:\\s*\\*{{0,2}}\\s*(.*)$",
                candidate,
                flags=re.IGNORECASE,
            )
            if match:
                matched_label = normalized_labels[label]
                inline_value = match.group(1).rstrip("*").strip()
                break
'''
    reasoning_old = '''                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    assistant_message["reasoning"] = reasoning

                if not tool_calls:
'''
    reasoning_new = '''                if reasoning:
                    captured_reasoning = reasoning
                    # Flash tool calls usually leave content empty, so retain a labelled
                    # world-model update from reasoning before the next analyzer turn.
                    self._update_summarized_knowledge_from_assistant(reasoning)
                    append_transcript("THINKING", reasoning)
                    assistant_message["reasoning"] = reasoning

                if not tool_calls:
'''

    for old, new in (
        (label_parser_old, label_parser_new),
        (reasoning_old, reasoning_new),
    ):
        if text.count(old) != 1:
            raise ValueError("Flash tool-agent patch anchor did not match exactly once.")
        text = text.replace(old, new, 1)

    compile(text, "flash_tool_agent.py", "exec")
    return text
