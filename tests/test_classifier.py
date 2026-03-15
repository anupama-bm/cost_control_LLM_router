"""
Run this to verify your classifier works before building the router.
Usage: python -m pytest tests/test_classifier.py -v -s
"""

import pytest
import asyncio
from app.classifier.prompt_classifier import PromptClassifier
from app.schemas.pydantic_models import DifficultyTag


@pytest.mark.asyncio
async def test_simple_prompt():
    classifier = PromptClassifier()
    result = await classifier.classify("Summarize this article in 3 bullet points.")
    print(f"\nTags: {result.tags}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    assert DifficultyTag.SIMPLE_SUMMARIZATION in result.tags
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_code_prompt():
    classifier = PromptClassifier()
    result = await classifier.classify(
        "Write a Python function that implements binary search on a sorted list."
    )
    print(f"\nTags: {result.tags}")
    print(f"Confidence: {result.confidence}")
    assert DifficultyTag.CODE_GENERATION in result.tags


@pytest.mark.asyncio
async def test_complex_prompt():
    classifier = PromptClassifier()
    result = await classifier.classify(
        """Analyze the macroeconomic implications of the Fed raising interest rates
        by 50 basis points in the context of current inflation trends, and model
        the second-order effects on emerging market debt."""
    )
    print(f"\nTags: {result.tags}")
    print(f"Confidence: {result.confidence}")
    primary = classifier.get_primary_tag(result)
    print(f"Primary tag: {primary}")
    assert primary in [
        DifficultyTag.HIGH_CONTEXT_ANALYSIS,
        DifficultyTag.MULTI_STEP_REASONING
    ]


@pytest.mark.asyncio
async def test_primary_tag_selection():
    """
    Tests that when multiple tags exist, we pick the hardest one.
    """
    classifier = PromptClassifier()
    from app.schemas.pydantic_models import ClassificationResult
    mock_result = ClassificationResult(
        tags=[
            DifficultyTag.SIMPLE_SUMMARIZATION,
            DifficultyTag.CODE_GENERATION
        ],
        confidence=0.75,
        reasoning="Test"
    )
    primary = classifier.get_primary_tag(mock_result)
    assert primary == DifficultyTag.CODE_GENERATION  # weight 3 > weight 1


## Interview Deep-Dives for This Step
"""
**Q: Why use an LLM to classify instead of a traditional ML classifier like BERT?**

> "Two reasons. First, we're already in an LLM infrastructure — adding a BERT model means another service to deploy, monitor, and maintain. Second, LLMs understand semantic nuance. BERT trained on our labels needs thousands of labeled examples. The LLM classifier works on day one with zero training data. The tradeoff is latency and cost — which is why we use the smallest, fastest model (8B parameters) for this job."

**Q: What happens if the classifier is down?**

> "The fallback classifier activates. It returns `high_context_analysis` with zero confidence. Zero confidence triggers maximum escalation in the router — we route to the strongest available model. The user gets a correct answer, we log the failure, and an alert fires. We never sacrifice user experience for internal failures."

**Q: Why `temperature=0.1` for the classifier?**

> "Classification is not a creative task. We want the same prompt to always get the same tag — determinism. High temperature introduces randomness which would make routing unpredictable. `0.1` isn't zero because some prompts are genuinely ambiguous and we want the model to express that via a lower confidence score rather than forcing a wrong high-confidence answer."

**Q: What is Tenacity and why use it?**

> "Tenacity is a Python retry library. LLM APIs fail transiently — network blips, rate limits, malformed outputs. Without retries, ~2% of requests would fail with a 500 error unnecessarily. Tenacity wraps our function with exponential backoff: wait 1 second, then 2, then 4 before giving up. This handles the vast majority of transient failures invisibly."
"""