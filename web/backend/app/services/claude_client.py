"""Claude API client for LLM-powered analyst."""

import os
from typing import Optional

import requests


def get_claude_response(
    system_prompt: str,
    user_content: str,
    model: Optional[str] = None,
) -> str:
    """
    Call Claude API to get assistant response.

    Args:
        system_prompt: System prompt for Claude
        user_content: User message content
        model: Model name (defaults to SPECTRA_LLM_MODEL env var or claude-sonnet-4-20250514)

    Returns:
        Assistant response text

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
        requests.HTTPError: If API call fails
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    model = model or os.getenv("SPECTRA_LLM_MODEL", "claude-sonnet-4-20250514")

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    # Extract assistant text from response
    if "content" in data and len(data["content"]) > 0:
        content_item = data["content"][0]
        if isinstance(content_item, dict) and "text" in content_item:
            return content_item["text"]
    raise ValueError("Unexpected response format from Claude API")

