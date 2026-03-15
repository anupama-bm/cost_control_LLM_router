"""
Fallback Classifier — used when the LLM classifier fails all retries.

Why do we need this?
In production, you CANNOT return a 500 error just because the classifier
had a bad day. The system must degrade gracefully.

Strategy: if classification fails, assume HIGH_CONTEXT_ANALYSIS with low
confidence. This routes to the strongest model (GPT-4o / best Groq model),
which is the safe choice. We'd rather overspend on one request than
return a bad response to a user.

Interview talking point:
"We designed for graceful degradation. The fallback classifier ensures
100% uptime for the routing pipeline. It logs the failure so engineers
are alerted, but the user never sees an error."
"""

import structlog
from app.schemas.pydantic_models import ClassificationResult, DifficultyTag

logger = structlog.get_logger(__name__)


def get_fallback_classification(prompt: str, error: Exception) -> ClassificationResult:
    """
    Called when PromptClassifier.classify() fails after all retries.

    Returns a safe default: assume hardest category, low confidence.
    Low confidence alone will trigger escalation to strongest model in router.
    """

    logger.error(
        "Classifier failed — using fallback classification",
        error=str(error),
        prompt_length=len(prompt),
        fallback_tag=DifficultyTag.HIGH_CONTEXT_ANALYSIS.value
    )

    return ClassificationResult(
        tags=[DifficultyTag.HIGH_CONTEXT_ANALYSIS],
        confidence=0.0,   # 0.0 confidence = maximum escalation in router
        reasoning=f"Fallback classification due to classifier error: {str(error)}"
    )