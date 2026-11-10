"""Digest-bound follow-up for parsed Python calls that execute no action."""

import hashlib


PATCH_NAME = "inspection-action-followup-v4"
SOURCE_SHA256 = "535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d"


def patch_tool_agent(source: bytes) -> str:
    """Ask for an action after an inspection-only Python tool call."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Flash tool-agent source digest changed; refusing to patch.")

    text = source.decode("utf-8")
    before_dispatch = (
        '                assistant_message["tool_calls"] = tool_calls\n'
        "                messages.append(assistant_message)\n"
        "\n"
        "                for tool_index, tool_call in enumerate(tool_calls):\n"
    )
    after_dispatch = (
        '                assistant_message["tool_calls"] = tool_calls\n'
        "                messages.append(assistant_message)\n"
        "                inspection_only_python_call = False\n"
        "\n"
        "                for tool_index, tool_call in enumerate(tool_calls):\n"
    )
    if text.count(before_dispatch) != 1:
        raise ValueError("Flash inspection-action patch dispatch anchor did not match exactly once.")
    text = text.replace(before_dispatch, after_dispatch, 1)

    before_result = (
        "                    if dispatch.step_executed:\n"
        "                        step_executed = True\n"
        "                    append_transcript(f\"TOOL RESULT: {tool_name}\", _render_tool_result_display(dispatch.content))\n"
    )
    after_result = (
        "                    if dispatch.step_executed:\n"
        "                        step_executed = True\n"
        "                    elif tool_name == \"python\":\n"
        "                        inspection_only_python_call = True\n"
        "                    append_transcript(f\"TOOL RESULT: {tool_name}\", _render_tool_result_display(dispatch.content))\n"
    )
    if text.count(before_result) != 1:
        raise ValueError("Flash inspection-action patch result anchor did not match exactly once.")
    text = text.replace(before_result, after_result, 1)

    before_control = (
        "        def control_yield_reason() -> str | None:\n"
        "            if should_stop is not None:\n"
        "                try:\n"
        "                    if should_stop():\n"
        "                        return \"stop_requested\"\n"
        "                except Exception as exc:\n"
        "                    log.warning(\"analyzer stop check failed at action %d: %s\", display_action_num, exc)\n"
        "            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:\n"
        "                return \"turn_time_budget\"\n"
        "            return None\n"
        "\n"
        "        try:\n"
    )
    after_control = (
        "        def control_yield_reason() -> str | None:\n"
        "            if should_stop is not None:\n"
        "                try:\n"
        "                    if should_stop():\n"
        "                        return \"stop_requested\"\n"
        "                except Exception as exc:\n"
        "                    log.warning(\"analyzer stop check failed at action %d: %s\", display_action_num, exc)\n"
        "            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:\n"
        "                return \"turn_time_budget\"\n"
        "            return None\n"
        "\n"
        "        def append_inspection_action_followup() -> None:\n"
        "            nonlocal turn_started_at\n"
        "            followup_prompt = (\n"
        "                \"Your last Python tool call inspected the state but did not execute a real environment action. \"\n"
        "                \"Use those observations now. On this bounded follow-up, emit exactly one `python` tool call \"\n"
        "                \"that calls `action(actions)` with the best valid action or short sequence. \"\n"
        "                \"Do not make another inspection-only call unless it is required to avoid an invalid or terminal action.\"\n"
        "            )\n"
        "            append_transcript(\"ANALYZER STATUS\", \"inspection_action_followup_forced\")\n"
        "            append_transcript(\"USER PROMPT\", followup_prompt)\n"
        "            messages.append({\"role\": \"user\", \"content\": followup_prompt})\n"
        "            turn_started_at = time.monotonic()\n"
        "\n"
        "        try:\n"
    )
    if text.count(before_control) != 1:
        raise ValueError("Flash inspection-action patch control anchor did not match exactly once.")
    text = text.replace(before_control, after_control, 1)

    before_no_tool_yield = (
        "                    if content or reasoning:\n"
        "                        messages.append(assistant_message)\n"
        "                    yielded_control_reason = control_yield_reason()\n"
    )
    after_no_tool_yield = (
        "                    if content or reasoning:\n"
        "                        messages.append(assistant_message)\n"
        "                    if forced_inspection_followup_used:\n"
        "                        yielded_control_reason = \"inspection_action_followup_exhausted\"\n"
        "                        break\n"
        "                    yielded_control_reason = control_yield_reason()\n"
    )
    if text.count(before_no_tool_yield) != 1:
        raise ValueError("Flash inspection-action patch no-tool anchor did not match exactly once.")
    text = text.replace(before_no_tool_yield, after_no_tool_yield, 1)

    before_loop_end = (
        "                if yielded_control_reason is not None:\n"
        "                    break\n"
        "                if step_executed:\n"
        "                    break\n"
        "\n"
        "        except requests.RequestException as exc:\n"
    )
    after_loop_end = (
        "                if yielded_control_reason is not None:\n"
        "                    if (\n"
        "                        yielded_control_reason == \"turn_time_budget\"\n"
        "                        and inspection_only_python_call\n"
        "                        and not forced_inspection_followup_used\n"
        "                    ):\n"
        "                        forced_inspection_followup_used = True\n"
        "                        append_inspection_action_followup()\n"
        "                        yielded_control_reason = None\n"
        "                        continue\n"
        "                    break\n"
        "                if step_executed:\n"
        "                    break\n"
        "                if forced_inspection_followup_used:\n"
        "                    yielded_control_reason = \"inspection_action_followup_exhausted\"\n"
        "                    break\n"
        "\n"
        "        except requests.RequestException as exc:\n"
    )
    if text.count(before_loop_end) != 1:
        raise ValueError("Flash inspection-action patch loop-end anchor did not match exactly once.")
    text = text.replace(before_loop_end, after_loop_end, 1)

    before_yield_state = (
        "        turn_started_at = time.monotonic()\n"
        "        yielded_control_reason: str | None = None\n"
    )
    after_yield_state = (
        "        turn_started_at = time.monotonic()\n"
        "        yielded_control_reason: str | None = None\n"
        "        forced_inspection_followup_used = False\n"
    )
    if text.count(before_yield_state) != 1:
        raise ValueError("Flash inspection-action patch state anchor did not match exactly once.")
    text = text.replace(before_yield_state, after_yield_state, 1)

    compile(text, "flash_tool_agent_inspection_action.py", "exec")
    return text
