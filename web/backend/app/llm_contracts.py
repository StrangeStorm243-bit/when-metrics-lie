"""Pydantic contracts for LLM API endpoints."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FocusContext(BaseModel):
    """Focus context for comparison explanation."""

    type: Literal["scenario", "component", "flag"]
    key: str


class CompareExplainRequest(BaseModel):
    """Request to explain a comparison using LLM."""

    intent: str = Field(
        ...,
        description="Analyst intent (overview, worse, improved, worst_case, new_flags, scenario_focus, component_focus, flag_focus)",
    )
    focus: Optional[FocusContext] = Field(None, description="Optional focus context")
    context: dict = Field(
        ..., description="Comparison context bundle (JSON-serializable)"
    )
    user_question: Optional[str] = Field(None, description="Optional user question")


class CompareExplainResponse(BaseModel):
    """Response from LLM explanation."""

    title: str = Field(..., description="Response title")
    body_markdown: str = Field(..., description="Response body in markdown")
    evidence_keys: list[str] = Field(
        default_factory=list,
        description="List of evidence keys referenced (scenario:<id>, component:<name>, flag:<code>)",
    )
