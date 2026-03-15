"""
POST /api/v1/feedback — human feedback endpoint.

Accepts ratings and flags on completed requests.
Triggers adaptive routing analysis after saving.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import FeedbackLog
from app.db.session import get_db
from app.db import crud
from app.schemas.pydantic_models import FeedbackRequest, FeedbackResponse
from app.feedback.adaptive_engine import AdaptiveRoutingEngine

logger = structlog.get_logger(__name__)
router = APIRouter()

adaptive_engine = AdaptiveRoutingEngine()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on a completion",
    description="""
    Submit human feedback on a routed completion.
    Rating 1-5, with optional underpowered/overkill flags.
    Feedback is used to improve future routing decisions.
    """,
)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """
    Feedback pipeline:
        1. Validate request_id exists
        2. Check no duplicate feedback
        3. Save feedback
        4. Run adaptive routing analysis
        5. Return result
    """

    # ── Validate request exists ───────────────────────────────────────────
    existing_request = await crud.get_request_by_id(db, request.request_id)
    if not existing_request:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request.request_id} not found",
        )

    # ── Check for duplicate feedback ──────────────────────────────────────
    # The DB has a unique constraint too, but we check here
    # to give a cleaner error message
    existing_feedback = await db.execute(
        select(FeedbackLog).where(FeedbackLog.request_id == request.request_id)
    )
    if existing_feedback.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Feedback already submitted for request {request.request_id}",
        )

    # ── Save feedback ─────────────────────────────────────────────────────
    feedback = await crud.save_feedback(
        db,
        request_id=request.request_id,
        rating=request.rating,
        underpowered=request.underpowered,
        overkill=request.overkill,
        comment=request.comment,
    )

    logger.info(
        "Feedback received",
        request_id=str(request.request_id),
        rating=request.rating,
        underpowered=request.underpowered,
        overkill=request.overkill,
    )

    # ── Run adaptive routing analysis ─────────────────────────────────────
    # Check if this feedback pushes any tag over the adjustment threshold
    adjustment_triggered = await adaptive_engine.analyze_and_adjust(
        db=db,
        tag=existing_request.difficulty_tags[0],
        model=existing_request.routed_model,
    )

    return FeedbackResponse(
        feedback_id=feedback.id,
        message="Feedback recorded successfully",
        routing_adjustment_triggered=adjustment_triggered,
    )