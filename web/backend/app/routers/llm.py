"""LLM API router for Claude-powered analyst."""

import json
import os
import re

import requests
from fastapi import APIRouter, HTTPException, status

from ..llm_contracts import CompareExplainRequest, CompareExplainResponse
from ..services.claude_client import get_claude_response

router = APIRouter(prefix="/llm", tags=["llm"])

SYSTEM_PROMPT = """You are a senior ML reliability analyst. You may ONLY use facts from the provided JSON context. If a claim is not supported by the context, say you are unsure. Do not speculate. Produce a concise answer with bullet points and include an 'Evidence:' section listing evidence keys present in the context (scenario:<id>, component:<name>, flag:<code>) when relevant."""


@router.post("/compare-explain", response_model=CompareExplainResponse)
async def compare_explain(request: CompareExplainRequest) -> CompareExplainResponse:
    """Explain a comparison using Claude LLM."""
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LLM features require ANTHROPIC_API_KEY environment variable to be set",
        )

    # Build user content
    context_json = json.dumps(request.context, indent=2)
    user_content_parts = [
        f"Intent: {request.intent}",
    ]

    if request.focus:
        user_content_parts.append(f"Focus: {request.focus.type}={request.focus.key}")

    if request.user_question:
        user_content_parts.append(f"User question: {request.user_question}")

    user_content_parts.append("\nContext (JSON):")
    user_content_parts.append(context_json)

    user_content = "\n".join(user_content_parts)

    try:
        # Call Claude
        assistant_text = get_claude_response(SYSTEM_PROMPT, user_content)

        # Parse response (simple: first line as title, rest as body)
        lines = assistant_text.strip().split("\n")
        title = lines[0] if lines else "AI Analysis"
        body_markdown = "\n".join(lines[1:]) if len(lines) > 1 else assistant_text

        # Extract evidence keys (simple pattern matching)
        evidence_keys = []
        for line in lines:
            if "scenario:" in line.lower():
                # Try to extract scenario IDs
                matches = re.findall(r"scenario:([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
                evidence_keys.extend([f"scenario:{m}" for m in matches])
            if "component:" in line.lower():
                matches = re.findall(r"component:([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
                evidence_keys.extend([f"component:{m}" for m in matches])
            if "flag:" in line.lower():
                matches = re.findall(r"flag:([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
                evidence_keys.extend([f"flag:{m}" for m in matches])

        # Also check context for evidence keys
        if request.focus:
            evidence_keys.append(f"{request.focus.type}:{request.focus.key}")

        return CompareExplainResponse(
            title=title,
            body_markdown=body_markdown,
            evidence_keys=list(set(evidence_keys)),  # Deduplicate
        )

    except ValueError as e:
        # API key missing (shouldn't happen due to check above, but handle gracefully)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )
    except requests.RequestException as e:
        # External API failure (network, timeout, etc.)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service unavailable",
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM request failed",
        )
