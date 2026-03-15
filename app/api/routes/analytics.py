"""
Analytics endpoints — cost reporting and system insights.

These endpoints answer the questions leadership asks:
    "How much are we saving?"
    "Which model handles most requests?"
    "What's our average response time?"
"""

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db import crud
from app.cost.cost_calculator import CostCalculator
from decimal import Decimal

logger = structlog.get_logger(__name__)
router = APIRouter()
cost_calculator = CostCalculator()


@router.get(
    "/analytics/cost-summary",
    summary="Get cost savings summary",
    description="Aggregated cost and savings data for the specified time window.",
)
async def get_cost_summary(
    lookback_days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """
    The executive dashboard endpoint.
    Returns the numbers that justify the platform.
    """
    summary = await crud.get_cost_summary(db, lookback_days=lookback_days)

    # Add monthly projection based on observed savings rate
    if summary["total_requests"] > 0:
        avg_saved_per_request = (
            summary["total_saved_usd"] / summary["total_requests"]
        )
        daily_request_rate = summary["total_requests"] / lookback_days
        projection = cost_calculator.project_monthly_savings(
            cost_saved_per_request=Decimal(str(avg_saved_per_request)),
            requests_per_day=int(daily_request_rate),
        )
        summary["projections"] = projection

    return summary


@router.get(
    "/analytics/recent-requests",
    summary="Get recent requests",
    description="Returns recent requests for monitoring and debugging.",
)
async def get_recent_requests(
    limit: int = Query(default=20, ge=1, le=100),
    caller_id: str = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    requests = await crud.get_recent_requests(
        db,
        limit=limit,
        caller_id=caller_id,
    )

    result = []
    for r in requests:
        # Fetch feedback for this request if it exists
        from sqlalchemy import select
        from app.db.models import FeedbackLog
        fb_result = await db.execute(
            select(FeedbackLog).where(FeedbackLog.request_id == r.id)
        )
        feedback = fb_result.scalar_one_or_none()

        result.append({
            "request_id": str(r.id),
            "timestamp": r.timestamp.isoformat(),
            "prompt": r.prompt,
            "routed_model": r.routed_model,
            "difficulty_tags": r.difficulty_tags,
            "confidence": r.classifier_confidence,
            "cost_saved_usd": float(r.cost_saved_usd),
            "latency_ms": r.latency_ms,
            "caller_id": r.caller_id,
            "feedback": {
                "rating": feedback.rating,
                "underpowered": feedback.underpowered,
                "overkill": feedback.overkill,
                "comment": feedback.comment,
            } if feedback else None,
        })

    return {"count": len(result), "requests": result}


@router.get(
    "/analytics/routing-table",
    summary="View current routing rules",
    description="Shows the active routing table including any adaptive overrides.",
)
async def get_routing_table(
    db: AsyncSession = Depends(get_db),
):
    """
    Full transparency into how the system is currently routing.
    Adaptive overrides show how feedback has changed routing.
    """
    from app.router.model_router import BASE_ROUTING_TABLE, explain_routing_table
    from app.config import get_settings

    settings = get_settings()
    adaptive_overrides = await crud.get_adaptive_overrides(db)

    base_rules = [
        {
            "tag": tag.value,
            "default_model": model.value,
            "currently_overridden": tag.value in adaptive_overrides,
            "active_model": adaptive_overrides.get(tag.value, model.value),
        }
        for tag, model in BASE_ROUTING_TABLE.items()
    ]

    return {
        "base_routing_table": base_rules,
        "adaptive_overrides": adaptive_overrides,
        "confidence_thresholds": {
            "low": settings.low_confidence_threshold,
            "high": settings.high_confidence_threshold,
        },
        "explanation": explain_routing_table(),
    }