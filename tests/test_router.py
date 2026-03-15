"""
Router tests — no database, no LLM calls needed.
The router is pure logic, so tests are instant and deterministic.

This is intentional design: by keeping the router stateless,
we can test every routing scenario in milliseconds.
"""

import pytest
from app.router.model_router import ModelRouter, escalate_model, demote_model
from app.schemas.pydantic_models import (
    ClassificationResult,
    DifficultyTag,
    ModelName,
)


router = ModelRouter()


def make_classification(tag: DifficultyTag, confidence: float) -> ClassificationResult:
    return ClassificationResult(
        tags=[tag],
        confidence=confidence,
        reasoning="test"
    )


# ── Base routing tests ────────────────────────────────────

def test_simple_routes_to_phi3():
    result = router.route(
        classification=make_classification(DifficultyTag.SIMPLE_SUMMARIZATION, 0.9),
        primary_tag=DifficultyTag.SIMPLE_SUMMARIZATION,
    )
    assert result.selected_model == ModelName.PHI3


def test_code_routes_to_llama():
    result = router.route(
        classification=make_classification(DifficultyTag.CODE_GENERATION, 0.85),
        primary_tag=DifficultyTag.CODE_GENERATION,
    )
    assert result.selected_model == ModelName.LLAMA3_70B


def test_high_context_routes_to_gpt4o():
    result = router.route(
        classification=make_classification(DifficultyTag.HIGH_CONTEXT_ANALYSIS, 0.9),
        primary_tag=DifficultyTag.HIGH_CONTEXT_ANALYSIS,
    )
    assert result.selected_model == ModelName.GPT4O


# ── Confidence escalation tests ───────────────────────────

def test_low_confidence_escalates_phi3_to_llama():
    """
    simple_summarization normally → PHI3
    but low confidence → escalate to LLAMA3_70B
    """
    result = router.route(
        classification=make_classification(DifficultyTag.SIMPLE_SUMMARIZATION, 0.3),
        primary_tag=DifficultyTag.SIMPLE_SUMMARIZATION,
    )
    assert result.selected_model == ModelName.LLAMA3_70B


def test_low_confidence_at_max_tier_stays():
    """
    high_context_analysis → GPT4O (max tier)
    low confidence → tries to escalate, already at max → stays GPT4O
    """
    result = router.route(
        classification=make_classification(DifficultyTag.HIGH_CONTEXT_ANALYSIS, 0.2),
        primary_tag=DifficultyTag.HIGH_CONTEXT_ANALYSIS,
    )
    assert result.selected_model == ModelName.GPT4O


# ── Adaptive override tests ───────────────────────────────

def test_adaptive_override_applied():
    """
    Feedback has taught us code_generation needs GPT4O.
    Override should take priority over base routing.
    """
    result = router.route(
        classification=make_classification(DifficultyTag.CODE_GENERATION, 0.9),
        primary_tag=DifficultyTag.CODE_GENERATION,
        adaptive_overrides={"code_generation": "gpt-4o"},
    )
    assert result.selected_model == ModelName.GPT4O


def test_adaptive_override_priority_over_confidence():
    """
    Even low confidence should not override adaptive rules.
    Adaptive rules are highest priority.
    """
    result = router.route(
        classification=make_classification(DifficultyTag.CODE_GENERATION, 0.2),
        primary_tag=DifficultyTag.CODE_GENERATION,
        adaptive_overrides={"code_generation": "gpt-4o"},
    )
    assert result.selected_model == ModelName.GPT4O


# ── Tier movement tests ───────────────────────────────────

def test_escalate_from_phi3():
    model, changed = escalate_model(ModelName.PHI3)
    assert model == ModelName.LLAMA3_70B
    assert changed is True


def test_escalate_from_max_tier():
    model, changed = escalate_model(ModelName.GPT4O)
    assert model == ModelName.GPT4O
    assert changed is False


def test_demote_from_gpt4o():
    model, changed = demote_model(ModelName.GPT4O)
    assert model == ModelName.LLAMA3_70B
    assert changed is True


def test_demote_from_min_tier():
    model, changed = demote_model(ModelName.PHI3)
    assert model == ModelName.PHI3
    assert changed is False