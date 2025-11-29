"""
This module defines data models used across the application, primarily utilizing Pydantic for data validation and serialization.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """Represents a template for generating prompts, including system instructions, user instructions, and context format."""

    system_instruction: str
    instructions: List[str]
    context_format: str


class RetrievalResult(BaseModel):
    """Represents a retrieval result, including content, metadata, and relevance score."""

    content: str
    metadata: dict
    relevance_score: float


class Message(BaseModel):
    """Represents a message, including content, role, timestamp, user ID, and optional metadata."""

    content: str

    role: Literal["user", "assistant"]

    timestamp: datetime
    user_id: str

    metadata: Optional[dict] = {}


class ConfidenceResult(BaseModel):
    """Represents a confidence result, including retrieval confidence, LLM confidence, overall confidence, and confidence breakdown."""

    retrieval_confidence: float
    llm_confidence: float
    overall_confidence: float
    is_confident: bool
    confidence_breakdown: dict


class ScoringWeights(BaseModel):
    """Represents the weights for scoring retrieval and LLM responses."""

    retrieval_weight: float = 0.4
    llm_weight: float = 0.6
