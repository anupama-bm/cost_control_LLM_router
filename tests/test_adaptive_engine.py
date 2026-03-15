"""
Adaptive engine tests.

We test the decision logic by mocking the DB stats.
This means we can test every threshold scenario without
needing actual feedback data in a real database.

Usage: python -m pytest tests/test_adaptive_engine.py -v -s
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.feedback.adaptive_engine import (
    AdaptiveRoutingEngine,
    UNDERPOWERED_THRESHOLD,
    OVERKILL_THRESHOLD,
    MIN_SAMPLE_SIZE,
)


engine = AdaptiveRoutingEngine()


@pytest.mark.asyncio
async def test_no_adjustment_insufficient_samples():
    """
    Below MIN_SAMPLE_SIZE → no adjustment regardless of rates.
    Prevents noise-driven routing changes.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 5,      # Below MIN_SAMPLE_SIZE (10)
            "avg_rating": 1.5,
            "underpowered_count": 4,
            "overkill_count": 0,
            "underpowered_rate": 0.80,  # Very high but sample too small
            "overkill_rate": 0.0,
        })

        result = await engine.analyze_and_adjust(
            db=mock_db,
            tag="code_generation",
            model="phi-3",
        )

    assert result is False  # No adjustment despite high underpowered rate
    print("\n✓ Correctly ignored high underpowered rate with small sample")


@pytest.mark.asyncio
async def test_escalation_triggered():
    """
    Sufficient samples + high underpowered rate → escalation.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 15,
            "avg_rating": 2.1,
            "underpowered_count": 6,
            "overkill_count": 0,
            "underpowered_rate": 0.40,  # Above UNDERPOWERED_THRESHOLD (0.30)
            "overkill_rate": 0.0,
        })
        mock_crud.save_routing_adjustment = AsyncMock()

        # Mock cooldown check — not in cooldown
        with patch.object(engine, "_check_cooldown", return_value=False):
            result = await engine.analyze_and_adjust(
                db=mock_db,
                tag="code_generation",
                model="phi-3",
            )

    assert result is True
    mock_crud.save_routing_adjustment.assert_called_once()

    call_kwargs = mock_crud.save_routing_adjustment.call_args.kwargs
    assert call_kwargs["difficulty_tag"] == "code_generation"
    assert call_kwargs["previous_model"] == "phi-3"
    assert call_kwargs["new_model"] == "llama-3-70b"

    print(f"\n✓ Escalation: phi-3 → llama-3-70b")
    print(f"  Reason: underpowered_rate={0.40:.1%} > threshold={UNDERPOWERED_THRESHOLD:.1%}")


@pytest.mark.asyncio
async def test_demotion_triggered():
    """
    Sufficient samples + high overkill rate → demotion.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 20,
            "avg_rating": 4.5,
            "underpowered_count": 0,
            "overkill_count": 10,
            "underpowered_rate": 0.0,
            "overkill_rate": 0.50,  # Above OVERKILL_THRESHOLD (0.40)
        })
        mock_crud.save_routing_adjustment = AsyncMock()

        with patch.object(engine, "_check_cooldown", return_value=False):
            result = await engine.analyze_and_adjust(
                db=mock_db,
                tag="simple_summarization",
                model="gpt-4o",
            )

    assert result is True
    call_kwargs = mock_crud.save_routing_adjustment.call_args.kwargs
    assert call_kwargs["previous_model"] == "gpt-4o"
    assert call_kwargs["new_model"] == "llama-3-70b"

    print(f"\n✓ Demotion: gpt-4o → llama-3-70b")
    print(f"  Reason: overkill_rate={0.50:.1%} > threshold={OVERKILL_THRESHOLD:.1%}")


@pytest.mark.asyncio
async def test_cooldown_prevents_adjustment():
    """
    Even with threshold exceeded, cooldown blocks adjustment.
    Prevents thrashing.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 15,
            "avg_rating": 1.8,
            "underpowered_count": 8,
            "overkill_count": 0,
            "underpowered_rate": 0.53,
            "overkill_rate": 0.0,
        })

        # Mock cooldown — still active
        with patch.object(engine, "_check_cooldown", return_value=True):
            result = await engine.analyze_and_adjust(
                db=mock_db,
                tag="multi_step_reasoning",
                model="phi-3",
            )

    assert result is False
    print("\n✓ Cooldown correctly blocked adjustment")


@pytest.mark.asyncio
async def test_no_adjustment_when_rates_normal():
    """
    Rates below both thresholds → no change.
    This is the happy path — routing is working well.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 25,
            "avg_rating": 4.2,
            "underpowered_count": 3,
            "overkill_count": 4,
            "underpowered_rate": 0.12,  # Below 0.30
            "overkill_rate": 0.16,      # Below 0.40
        })

        with patch.object(engine, "_check_cooldown", return_value=False):
            result = await engine.analyze_and_adjust(
                db=mock_db,
                tag="information_extraction",
                model="phi-3",
            )

    assert result is False
    print("\n✓ No adjustment needed — routing performing well")
    print(f"  underpowered_rate=12% < threshold={UNDERPOWERED_THRESHOLD:.0%}")
    print(f"  overkill_rate=16% < threshold={OVERKILL_THRESHOLD:.0%}")


@pytest.mark.asyncio
async def test_cannot_escalate_beyond_max_tier():
    """
    If already at GPT4O (max tier), escalation is rejected.
    """
    mock_db = AsyncMock()

    with patch("app.feedback.adaptive_engine.crud") as mock_crud:
        mock_crud.get_feedback_stats_for_tag = AsyncMock(return_value={
            "total_feedback": 15,
            "avg_rating": 2.0,
            "underpowered_count": 8,
            "overkill_count": 0,
            "underpowered_rate": 0.53,
            "overkill_rate": 0.0,
        })
        mock_crud.save_routing_adjustment = AsyncMock()

        with patch.object(engine, "_check_cooldown", return_value=False):
            result = await engine.analyze_and_adjust(
                db=mock_db,
                tag="high_context_analysis",
                model="gpt-4o",  # Already at max
            )

    # Should return False — can't go higher than max tier
    assert result is False
    mock_crud.save_routing_adjustment.assert_not_called()
    print("\n✓ Correctly rejected escalation beyond max tier")
