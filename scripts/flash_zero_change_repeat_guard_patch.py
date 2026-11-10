"""Digest-bound guard for repeated singleton no-change Flash actions."""

import hashlib


PATCH_NAME = "zero-change-repeat-guard-v1"
SOURCE_SHA256 = "535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d"


def patch_tool_agent(source: bytes) -> str:
    """Block one immediate singleton repeat after a confirmed no-change action."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Flash tool-agent source digest changed; refusing to patch.")

    text = source.decode("utf-8")
    anchor = (
        "        terminal_action_result: dict[str, Any] | None = None\n"
        "\n"
        "        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:\n"
    )
    replacement = (
        "        terminal_action_result: dict[str, Any] | None = None\n"
        "\n"
        "        def _repeat_guard_display(action: dict[str, Any]) -> str:\n"
        "            action_name = str(action.get(\"action\", \"\")).strip().upper()\n"
        "            model_name = to_model_action(action_name)\n"
        "            if model_name == \"MOUSE\":\n"
        "                return (\n"
        "                    f\"MOUSE(row={action.get('row')}, col={action.get('col')})\"\n"
        "                )\n"
        "            return model_name\n"
        "\n"
        "        def _should_block_zero_change_repeat(\n"
        "            actions: list[dict[str, Any]],\n"
        "        ) -> bool:\n"
        "            if len(actions) != 1 or not isinstance(self._last_action_result, dict):\n"
        "                return False\n"
        "            previous = self._last_action_result\n"
        "            if not previous.get(\"executed\") or previous.get(\"board_changed\") is not False:\n"
        "                return False\n"
        "            if any(\n"
        "                bool(previous.get(key))\n"
        "                for key in (\"done\", \"level_completed\", \"game_over\", \"run_complete\")\n"
        "            ):\n"
        "                return False\n"
        "            try:\n"
        "                if float(previous.get(\"reward\", 0.0) or 0.0) != 0.0:\n"
        "                    return False\n"
        "            except (TypeError, ValueError):\n"
        "                return False\n"
        "            previous_actions = previous.get(\"executed_actions\")\n"
        "            if not isinstance(previous_actions, list) or len(previous_actions) != 1:\n"
        "                return False\n"
        "            if str(previous_actions[0]).strip() != _repeat_guard_display(actions[0]):\n"
        "                return False\n"
        "            previous_valid = previous.get(\"valid_actions\")\n"
        "            if not isinstance(previous_valid, list):\n"
        "                return False\n"
        "            return _normalize_valid_actions(previous_valid) == list(self._current_valid_actions)\n"
        "\n"
        "        def _zero_change_repeat_payload(action: dict[str, Any]) -> dict[str, Any]:\n"
        "            previous = self._last_action_result or {}\n"
        "            display = _repeat_guard_display(action)\n"
        "            return {\n"
        "                \"executed\": False,\n"
        "                \"action_num\": previous.get(\"action_num\"),\n"
        "                \"level\": previous.get(\"level\"),\n"
        "                \"score\": previous.get(\"score\"),\n"
        "                \"reward\": 0.0,\n"
        "                \"state\": previous.get(\"state\"),\n"
        "                \"valid_actions\": list(self._current_valid_actions),\n"
        "                \"board_changed\": False,\n"
        "                \"done\": False,\n"
        "                \"level_completed\": False,\n"
        "                \"game_over\": False,\n"
        "                \"run_complete\": False,\n"
        "                \"action_display\": display,\n"
        "                \"requested_count\": 1,\n"
        "                \"executed_count\": 0,\n"
        "                \"stopped_early\": True,\n"
        "                \"guarded_repeat\": True,\n"
        "                \"stop_reason\": \"repeat_no_change_guard\",\n"
        "                \"stop_detail\": (\n"
        "                    \"Blocked one singleton action repeated immediately after it produced \"\n"
        "                    \"no board change, reward, or valid-action change. Choose a different \"\n"
        "                    \"action or inspect the current state.\"\n"
        "                ),\n"
        "            }\n"
        "\n"
        "        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("Flash zero-change guard anchor did not match exactly once.")
    text = text.replace(anchor, replacement, 1)

    dispatch_anchor = (
        "            normalized_actions = self._normalize_python_actions(actions)\n"
        "            if terminal_action_result is not None:\n"
    )
    dispatch_replacement = (
        "            normalized_actions = self._normalize_python_actions(actions)\n"
        "            if _should_block_zero_change_repeat(normalized_actions):\n"
        "                compact_payload = _zero_change_repeat_payload(normalized_actions[0])\n"
        "                self._last_action_result = dict(compact_payload)\n"
        "                return {\n"
        "                    \"action_result\": compact_payload,\n"
        "                    \"state\": _serialized_runtime_state(\n"
        "                        last_action_result=compact_payload,\n"
        "                    ),\n"
        "                }\n"
        "            if terminal_action_result is not None:\n"
    )
    if text.count(dispatch_anchor) != 1:
        raise ValueError("Flash zero-change guard dispatch anchor did not match exactly once.")
    text = text.replace(dispatch_anchor, dispatch_replacement, 1)

    compile(text, "flash_tool_agent_zero_change_repeat_guard.py", "exec")
    return text
