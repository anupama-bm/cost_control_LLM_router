
#CRUD Operations — the only file that speaks SQL.

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List
import structlog

from sqlalchemy import select, func, and_, desc, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RequestLog, FeedbackLog, RoutingAdjustment
from app.schemas.pydantic_models import (
    ModelName,
    DifficultyTag,
)

logger = structlog.get_logger(__name__)



async def log_request(
    db: AsyncSession,
    *,
    request_id: uuid.UUID,
    caller_id: Optional[str],
    prompt: str,
    difficulty_tags: List[str],
    classifier_confidence: float,
    routed_model: str,
    routing_reason: str,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    actual_cost_usd: Decimal,
    baseline_gpt4o_cost_usd: Decimal,
    cost_saved_usd: Decimal,
    latency_ms: float,
) -> RequestLog:

    db_request = RequestLog(
        id=request_id,
        caller_id=caller_id,
        prompt=prompt,
        difficulty_tags=difficulty_tags,
        classifier_confidence=classifier_confidence,
        routed_model=routed_model,
        routing_reason=routing_reason,
        response_text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        actual_cost_usd=actual_cost_usd,
        baseline_gpt4o_cost_usd=baseline_gpt4o_cost_usd,
        cost_saved_usd=cost_saved_usd,
        latency_ms=latency_ms,
    )

    db.add(db_request)
    await db.flush()  # Write to DB but don't commit yet
                      # Commit happens in get_db() dependency
                      # This gives us transaction safety —
                      # if anything fails after this, it rolls back

    logger.info(
        "Request logged to DB",
        request_id=str(request_id),
        routed_model=routed_model,
        cost_saved=str(cost_saved_usd),
    )

    return db_request


async def get_request_by_id(
    db: AsyncSession,
    request_id: uuid.UUID,
) -> Optional[RequestLog]:

    result = await db.execute(
        select(RequestLog).where(RequestLog.id == request_id)
    )
    return result.scalar_one_or_none()


# ═══════════════════════════════════════════════════════════
# FEEDBACK LOG OPERATIONS
# ═══════════════════════════════════════════════════════════

async def save_feedback(
    db: AsyncSession,
    *,
    request_id: uuid.UUID,
    rating: int,
    underpowered: bool,
    overkill: bool,
    comment: Optional[str],
) -> FeedbackLog:

    feedback = FeedbackLog(
        id=uuid.uuid4(),
        request_id=request_id,
        rating=rating,
        underpowered=underpowered,
        overkill=overkill,
        comment=comment,
    )

    db.add(feedback)
    await db.flush()

    logger.info(
        "Feedback saved",
        request_id=str(request_id),
        rating=rating,
        underpowered=underpowered,
        overkill=overkill,
    )

    return feedback


# ═══════════════════════════════════════════════════════════
# ADAPTIVE ROUTING OPERATIONS
# These are the most important reads in the system —
# called on EVERY request to check for routing overrides.
# ═══════════════════════════════════════════════════════════

async def get_adaptive_overrides(
    db: AsyncSession,
) -> Dict[str, str]:


    # Get the most recent adjustment per difficulty tag
    # Subquery: for each tag, find the max (latest) timestamp
    subquery = (
        select(
            RoutingAdjustment.difficulty_tag,
            func.max(RoutingAdjustment.timestamp).label("latest_ts"),
        )
        .group_by(RoutingAdjustment.difficulty_tag)
        .subquery()
    )

    # Join back to get the full row for each latest adjustment
    result = await db.execute(
        select(RoutingAdjustment).join(
            subquery,
            and_(
                RoutingAdjustment.difficulty_tag == subquery.c.difficulty_tag,
                RoutingAdjustment.timestamp == subquery.c.latest_ts,
            ),
        )
    )

    adjustments = result.scalars().all()

    # Convert to plain dict for the router
    overrides = {
        adj.difficulty_tag: adj.new_model
        for adj in adjustments
    }

    if overrides:
        logger.info("Adaptive overrides loaded", overrides=overrides)

    return overrides


async def save_routing_adjustment(
    db: AsyncSession,
    *,
    difficulty_tag: str,
    previous_model: str,
    new_model: str,
    reason: str,
    trigger_feedback_count: int,
    underpowered_rate: Optional[float] = None,
    overkill_rate: Optional[float] = None,
) -> RoutingAdjustment:
    """
    Records when the system changes its routing rule for a tag.
    This is the audit trail for adaptive routing decisions.

    Every time the feedback loop triggers a model tier change,
    this is called. Over time this table tells the story of
    how the system learned.
    """
    adjustment = RoutingAdjustment(
        id=uuid.uuid4(),
        difficulty_tag=difficulty_tag,
        previous_model=previous_model,
        new_model=new_model,
        reason=reason,
        trigger_feedback_count=trigger_feedback_count,
        underpowered_rate=underpowered_rate,
        overkill_rate=overkill_rate,
    )

    db.add(adjustment)
    await db.flush()

    logger.info(
        "Routing adjustment saved",
        tag=difficulty_tag,
        from_model=previous_model,
        to_model=new_model,
        reason=reason,
    )

    return adjustment


# ═══════════════════════════════════════════════════════════
# FEEDBACK ANALYSIS OPERATIONS
# Used by the adaptive routing engine to decide when
# to change routing rules.
# ═══════════════════════════════════════════════════════════

async def get_feedback_stats_for_tag(
    db: AsyncSession,
    tag: str,
    model: str,
    lookback_days: int = 7,
) -> Dict:
    """
    Computes feedback statistics for a specific tag+model combination
    over the last N days.

    This is what the adaptive routing engine queries to decide
    if routing rules need to change.

    Returns:
        {
            "total_feedback": 45,
            "avg_rating": 2.8,
            "underpowered_count": 18,
            "overkill_count": 3,
            "underpowered_rate": 0.40,
            "overkill_rate": 0.067,
        }

    Interview talking point:
        "We look back 7 days by default — recent enough to catch
        model degradation quickly, long enough to avoid adjusting
        on statistical noise. This is configurable. A high-traffic
        system might use 1 day; a low-traffic system might need 30."
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # Join feedback to requests to filter by tag and model
    result = await db.execute(
        select(
            func.count(FeedbackLog.id).label("total"),
            func.avg(FeedbackLog.rating).label("avg_rating"),
            func.sum(
                func.cast(FeedbackLog.underpowered, Integer)
            ).label("underpowered_count"),
            func.sum(
                func.cast(FeedbackLog.overkill, Integer)
            ).label("overkill_count"),
        )
        .join(RequestLog, FeedbackLog.request_id == RequestLog.id)
        .where(
            and_(
                # Filter to this specific tag+model combination
                RequestLog.routed_model == model,
                FeedbackLog.timestamp >= cutoff,
                # PostgreSQL array contains operator
                RequestLog.difficulty_tags.any(tag),
            )
        )
    )

    row = result.one()
    total = row.total or 0

    if total == 0:
        return {
            "total_feedback": 0,
            "avg_rating": None,
            "underpowered_count": 0,
            "overkill_count": 0,
            "underpowered_rate": 0.0,
            "overkill_rate": 0.0,
        }

    underpowered_count = int(row.underpowered_count or 0)
    overkill_count = int(row.overkill_count or 0)

    return {
        "total_feedback": total,
        "avg_rating": round(float(row.avg_rating), 2),
        "underpowered_count": underpowered_count,
        "overkill_count": overkill_count,
        "underpowered_rate": round(underpowered_count / total, 3),
        "overkill_rate": round(overkill_count / total, 3),
    }


# ═══════════════════════════════════════════════════════════
# ANALYTICS & REPORTING OPERATIONS
# Used for dashboards and cost reporting
# ═══════════════════════════════════════════════════════════

async def get_cost_summary(
    db: AsyncSession,
    lookback_days: int = 30,
) -> Dict:
    """
    Aggregated cost summary for the last N days.
    This is what powers a cost dashboard.

    Returns the numbers that justify the platform to leadership:
        - Total actual spend
        - Total baseline spend (what GPT-4o would have cost)
        - Total saved
        - Savings percentage
        - Breakdown by model
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    result = await db.execute(
        select(
            func.count(RequestLog.id).label("total_requests"),
            func.sum(RequestLog.actual_cost_usd).label("total_actual_cost"),
            func.sum(RequestLog.baseline_gpt4o_cost_usd).label("total_baseline_cost"),
            func.sum(RequestLog.cost_saved_usd).label("total_saved"),
            func.avg(RequestLog.latency_ms).label("avg_latency_ms"),
        )
        .where(RequestLog.timestamp >= cutoff)
    )

    row = result.one()

    total_actual = float(row.total_actual_cost or 0)
    total_baseline = float(row.total_baseline_cost or 0)
    total_saved = float(row.total_saved or 0)

    savings_pct = (
        round((total_saved / total_baseline) * 100, 1)
        if total_baseline > 0 else 0.0
    )

    # Breakdown by model — how often did each tier get used?
    model_result = await db.execute(
        select(
            RequestLog.routed_model,
            func.count(RequestLog.id).label("request_count"),
            func.sum(RequestLog.actual_cost_usd).label("model_cost"),
        )
        .where(RequestLog.timestamp >= cutoff)
        .group_by(RequestLog.routed_model)
        .order_by(desc("request_count"))
    )

    model_breakdown = [
        {
            "model": row.routed_model,
            "request_count": row.request_count,
            "total_cost_usd": round(float(row.model_cost or 0), 6),
        }
        for row in model_result.all()
    ]

    return {
        "lookback_days": lookback_days,
        "total_requests": row.total_requests or 0,
        "total_actual_cost_usd": round(total_actual, 6),
        "total_baseline_cost_usd": round(total_baseline, 6),
        "total_saved_usd": round(total_saved, 6),
        "savings_percentage": savings_pct,
        "avg_latency_ms": round(float(row.avg_latency_ms or 0), 1),
        "model_breakdown": model_breakdown,
    }


async def get_recent_requests(
    db: AsyncSession,
    limit: int = 20,
    caller_id: Optional[str] = None,
) -> List[RequestLog]:
    """
    Fetches recent requests for debugging and monitoring.
    Optionally filtered by caller_id for per-service analytics.
    """
    query = (
        select(RequestLog)
        .order_by(desc(RequestLog.timestamp))
        .limit(limit)
    )

    if caller_id:
        query = query.where(RequestLog.caller_id == caller_id)

    result = await db.execute(query)
    return result.scalars().all()