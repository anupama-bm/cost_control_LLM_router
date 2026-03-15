"""
Pydantic models define the API contract.
These are NOT database models — they are the shapes of data
entering and leaving the API boundary.

Rule: validate at the edges, trust internally.
"""

from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from enum import Enum
import uuid
from datetime import datetime


# ─────────────────────────────────────────────
# ENUMS — finite valid values, enforced at parse time
# ─────────────────────────────────────────────

class DifficultyTag(str, Enum):
    """
    Every prompt gets exactly one of these tags.
    Adding a new tag here automatically propagates to docs and validation.
    """
    SIMPLE_SUMMARIZATION = "simple_summarization"
    INFORMATION_EXTRACTION = "information_extraction"
    CODE_GENERATION = "code_generation"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    HIGH_CONTEXT_ANALYSIS = "high_context_analysis"


class ModelName(str, Enum):
    """
    The three tiers. New models = add here + add client.
    """
    PHI3 = "phi-3"
    LLAMA3_70B = "llama-3-70b"
    GPT4O = "gpt-4o"


# ─────────────────────────────────────────────
# INBOUND — what the developer sends us
# ─────────────────────────────────────────────

class CompletionRequest(BaseModel):
    """
    The only thing a developer needs to send.
    No model selection. No complexity hints. Just the prompt.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32000,
        description="The prompt to be routed and completed."
    )
    # Optional: caller can tag their system for cost attribution
    caller_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the calling service (e.g. 'search-api', 'chatbot-v2')"
    )


# ─────────────────────────────────────────────
# INTERNAL — data passed between pipeline stages
# ─────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """
    Output of the classifier. Fed into the router.
    """
    tags: List[DifficultyTag]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None  # Why did the classifier decide this?


class RoutingDecision(BaseModel):
    """
    Output of the router. Fed into the LLM executor.
    """
    selected_model: ModelName
    reason: str  # Human-readable explanation — critical for debugging


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class CostBreakdown(BaseModel):
    """
    Full cost transparency per request.
    """
    actual_cost_usd: float = Field(..., description="What we actually paid")
    baseline_gpt4o_cost_usd: float = Field(..., description="What GPT-4o would have cost")
    cost_saved_usd: float = Field(..., description="Savings vs always using GPT-4o")
    routed_model: ModelName


# ─────────────────────────────────────────────
# OUTBOUND — what the developer receives back
# ─────────────────────────────────────────────

class CompletionResponse(BaseModel):
    """
    Everything the developer gets back.
    Transparency is a feature — they can see why routing happened.
    """
    request_id: uuid.UUID
    response: str
    routed_model: ModelName
    difficulty_tags: List[DifficultyTag]
    classifier_confidence: float
    token_usage: TokenUsage
    cost_breakdown: CostBreakdown
    latency_ms: float


# ─────────────────────────────────────────────
# FEEDBACK — human rating after the fact
# ─────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """
    A human tells us whether the routing decision was good.
    request_id links back to the original CompletionRequest.
    """
    request_id: uuid.UUID = Field(..., description="ID from the original completion response")
    rating: int = Field(..., ge=1, le=5, description="1=terrible, 5=perfect")
    underpowered: bool = Field(
        default=False,
        description="True if the model was too weak — response was poor quality"
    )
    overkill: bool = Field(
        default=False,
        description="True if a cheaper model would have been fine"
    )
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: uuid.UUID
    message: str
    routing_adjustment_triggered: bool  # Did this feedback change the routing rules?