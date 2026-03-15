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
