"""harness/call_model.py — CLI wrapper called by agentic_loop.yml.

Usage:
    python harness/call_model.py --context-file <path>

Reads environment variables:
    MODEL_PROVIDER   : "openai" | "anthropic" | "google"
    MODEL_ID         : model identifier string (e.g. "gpt-4o", "claude-3-5-sonnet-20241022")
    MODEL_API_KEY    : provider API key

The context file must contain two sections separated by the line:
    ---USER---

Everything before that line is treated as the system prompt (prompt_context.md content).
Everything after is the user message (dynamic context).

Prints the parsed JSON response dict to stdout.
Exits with code 1 and prints error to stderr on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_context_file(path: str) -> tuple[str, str]:
    """Split context file into (system_prompt, user_message) at the ---USER--- separator."""
    content = Path(path).read_text(encoding="utf-8")
    separator = "\n---USER---\n"
    if separator in content:
        system_prompt, user_message = content.split(separator, 1)
    else:
        # No separator — treat entire content as user message
        system_prompt = ""
        user_message = content
    return system_prompt.strip(), user_message.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the configured LLM provider.")
    parser.add_argument("--context-file", required=True,
                        help="Path to the combined context file.")
    args = parser.parse_args()

    # Read required environment variables
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    model_id = os.environ.get("MODEL_ID", "").strip()
    api_key = os.environ.get("MODEL_API_KEY", "").strip()

    if not provider:
        print("ERROR: MODEL_PROVIDER environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not model_id:
        print("ERROR: MODEL_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("ERROR: MODEL_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Import adapters (done here so import errors surface clearly)
    try:
        from harness.providers import ADAPTERS  # noqa: PLC0415
    except ImportError:
        # Support running from repo root where harness/ is a plain directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from harness.providers import ADAPTERS  # noqa: PLC0415

    if provider not in ADAPTERS:
        supported = ", ".join(sorted(ADAPTERS.keys()))
        print(
            f"ERROR: Unknown MODEL_PROVIDER '{provider}'. Supported: {supported}",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = ADAPTERS[provider]
    system_prompt, user_message = _parse_context_file(args.context_file)

    try:
        result = adapter(
            model_id=model_id,
            api_key=api_key,
            system_prompt=system_prompt,
            user_message=user_message,
        )
    except ValueError as exc:
        print(f"ERROR: Provider returned invalid response.\n{exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
