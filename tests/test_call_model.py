"""tests/test_call_model.py — unit tests for harness/call_model.py.

Tests the CLI entry point (main()) by patching env vars and the provider
adapter, without making real API calls.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.call_model import _parse_context_file, main


_VALID_RESPONSE = {
    "thoughts": "Testing the call_model CLI.",
    "file_changes": [],
    "commands": ["/run-backtest"],
}

_SYSTEM = "You are a quant researcher."
_USER   = "Here is your context."


# ---------------------------------------------------------------------------
# _parse_context_file
# ---------------------------------------------------------------------------

class TestParseContextFile:
    def _write(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_splits_at_separator(self):
        path = self._write(f"{_SYSTEM}\n---USER---\n{_USER}")
        system, user = _parse_context_file(path)
        assert system == _SYSTEM
        assert user == _USER

    def test_no_separator_entire_content_is_user(self):
        path = self._write("all user content")
        system, user = _parse_context_file(path)
        assert system == ""
        assert user == "all user content"

    def test_strips_whitespace(self):
        path = self._write(f"  {_SYSTEM}  \n---USER---\n  {_USER}  ")
        system, user = _parse_context_file(path)
        assert system == _SYSTEM.strip()
        assert user == _USER.strip()

    def test_multiline_system_and_user(self):
        path = self._write("line1\nline2\n---USER---\nuline1\nuline2")
        system, user = _parse_context_file(path)
        assert "line1" in system
        assert "line2" in system
        assert "uline1" in user


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMain:
    def _context_file(self) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(f"{_SYSTEM}\n---USER---\n{_USER}")
        f.close()
        return f.name

    def _base_env(self, **overrides) -> dict:
        env = {
            "MODEL_PROVIDER": "openai",
            "MODEL_ID": "gpt-4o",
            "MODEL_API_KEY": "sk-fake",
        }
        env.update(overrides)
        return env

    def _run_main(self, env: dict, argv_extra: list[str]) -> tuple[str, int]:
        ctx = self._context_file()
        argv = ["call_model.py", "--context-file", ctx] + argv_extra
        captured = StringIO()
        exit_code = 0
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "argv", argv):
                with patch("sys.stdout", captured):
                    try:
                        main()
                    except SystemExit as e:
                        exit_code = e.code or 0
        return captured.getvalue(), exit_code

    def test_success_prints_json_to_stdout(self):
        fake_adapter = MagicMock(return_value=_VALID_RESPONSE)
        with patch.dict("harness.providers.ADAPTERS", {"openai": fake_adapter}):
            output, code = self._run_main(self._base_env(), [])
        assert code == 0
        parsed = json.loads(output)
        assert parsed["commands"] == ["/run-backtest"]

    def test_missing_model_provider_exits_1(self):
        env = self._base_env()
        del env["MODEL_PROVIDER"]
        _, code = self._run_main(env, [])
        assert code == 1

    def test_missing_model_id_exits_1(self):
        env = self._base_env()
        del env["MODEL_ID"]
        _, code = self._run_main(env, [])
        assert code == 1

    def test_missing_api_key_exits_1(self):
        env = self._base_env()
        del env["MODEL_API_KEY"]
        _, code = self._run_main(env, [])
        assert code == 1

    def test_unknown_provider_exits_1(self):
        env = self._base_env(MODEL_PROVIDER="unknown_llm")
        _, code = self._run_main(env, [])
        assert code == 1

    def test_adapter_value_error_exits_1(self):
        fake_adapter = MagicMock(side_effect=ValueError("malformed response: ..."))
        with patch.dict("harness.providers.ADAPTERS", {"openai": fake_adapter}):
            _, code = self._run_main(self._base_env(), [])
        assert code == 1

    def test_adapter_receives_correct_model_id(self):
        captured_args = {}

        def fake_adapter(model_id, api_key, system_prompt, user_message):
            captured_args.update({"model_id": model_id, "api_key": api_key})
            return _VALID_RESPONSE

        with patch.dict("harness.providers.ADAPTERS", {"openai": fake_adapter}):
            self._run_main(self._base_env(MODEL_ID="gpt-4-turbo"), [])

        assert captured_args["model_id"] == "gpt-4-turbo"

    def test_adapter_receives_api_key_from_env(self):
        captured_args = {}

        def fake_adapter(model_id, api_key, system_prompt, user_message):
            captured_args["api_key"] = api_key
            return _VALID_RESPONSE

        with patch.dict("harness.providers.ADAPTERS", {"openai": fake_adapter}):
            self._run_main(self._base_env(MODEL_API_KEY="my-secret-key"), [])

        assert captured_args["api_key"] == "my-secret-key"

    def test_system_prompt_comes_from_context_file(self):
        captured_args = {}

        def fake_adapter(model_id, api_key, system_prompt, user_message):
            captured_args["system_prompt"] = system_prompt
            return _VALID_RESPONSE

        with patch.dict("harness.providers.ADAPTERS", {"openai": fake_adapter}):
            self._run_main(self._base_env(), [])

        assert _SYSTEM.strip() in captured_args["system_prompt"]

    def test_anthropic_provider_uses_anthropic_adapter(self):
        fake_adapter = MagicMock(return_value=_VALID_RESPONSE)
        with patch.dict("harness.providers.ADAPTERS", {"anthropic": fake_adapter}):
            output, code = self._run_main(
                self._base_env(MODEL_PROVIDER="anthropic"), []
            )
        assert code == 0
        fake_adapter.assert_called_once()
