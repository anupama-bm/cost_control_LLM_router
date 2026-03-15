"""
POST /api/v1/complete — the main endpoint.

This is the ONLY endpoint developers interact with.
It orchestrates the entire routing pipeline.

Flow:
    Request → Classify → Load Overrides → Route →
    Call LLM → Calculate Cost → Log → Respond
"""

import uuid
import time
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db import crud
from app.classifier.prompt_classifier import PromptClassifier
from app.classifier.fallback_classifier import get_fallback_classification
from app.router.model_router import ModelRouter
from app.llm_clients.client_factory import get_client
from app.llm_clients.base import LLMClientError
from app.cost.cost_calculator import CostCalculator
from app.schemas.pydantic_models import (
    CompletionRequest,
    CompletionResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────
# SINGLETONS — instantiated once, reused across requests
# These are stateless objects — safe to share.
# ─────────────────────────────────────────────────────────

classifier = PromptClassifier()
model_router = ModelRouter()
cost_calculator = CostCalculator()


@router.post(
    "/complete",
    response_model=CompletionResponse,
    summary="Route and complete a prompt",
    description="""
    Accepts a prompt and automatically routes it to the optimal
    LLM based on complexity analysis and historical feedback.
    Returns the response with full cost transparency.
    """,
)
async def complete(
    request: CompletionRequest,
    db: AsyncSession = Depends(get_db),
) -> CompletionResponse:
    """
    Main completion endpoint — orchestrates the full pipeline.

    Depends(get_db) injects an async database session.
    FastAPI handles session lifecycle (commit/rollback)
    through the get_db() generator in session.py.
    """

    # Generate request ID early — used in logs throughout pipeline
    # so we can trace one request across all log lines
    request_id = uuid.uuid4()
    pipeline_start = time.perf_counter()

    logger.info(
        "Completion request received",
        request_id=str(request_id),
        caller_id=request.caller_id,
        prompt_length=len(request.prompt),
    )

    # ── Step 1: Classify the prompt ───────────────────────────────────────
    try:
        classification = await classifier.classify(request.prompt)
    except Exception as e:
        # Classifier failed all retries — use safe fallback
        logger.warning(
            "Classifier failed, using fallback",
            request_id=str(request_id),
            error=str(e),
        )
        classification = get_fallback_classification(request.prompt, e)

    primary_tag = classifier.get_primary_tag(classification)

    logger.info(
        "Prompt classified",
        request_id=str(request_id),
        primary_tag=primary_tag.value,
        confidence=classification.confidence,
        all_tags=[t.value for t in classification.tags],
    )

    # ── Step 2: Load adaptive routing overrides from DB ───────────────────
    # These are learned from historical feedback.
    # Empty dict on first run (no feedback yet).
    try:
        adaptive_overrides = await crud.get_adaptive_overrides(db)
    except Exception as e:
        # DB read failed — continue with no overrides (safe degradation)
        logger.warning(
            "Failed to load adaptive overrides, proceeding without",
            error=str(e),
        )
        adaptive_overrides = {}

    # ── Step 3: Route to optimal model ────────────────────────────────────
    routing_decision = model_router.route(
        classification=classification,
        primary_tag=primary_tag,
        adaptive_overrides=adaptive_overrides,
    )

    logger.info(
        "Routing decision made",
        request_id=str(request_id),
        selected_model=routing_decision.selected_model.value,
        reason=routing_decision.reason,
    )

    # ── Step 4: Call the selected LLM ─────────────────────────────────────
    try:
        llm_client = get_client(routing_decision.selected_model)
        llm_response = await llm_client.complete_with_timing(
            prompt=request.prompt,
            max_tokens=1024,
            temperature=0.7,
        )
    except LLMClientError as e:
        logger.error(
            "LLM call failed",
            request_id=str(request_id),
            model=routing_decision.selected_model.value,
            error=str(e),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LLM provider error",
                "model": routing_decision.selected_model.value,
                "message": str(e),
                "request_id": str(request_id),
            },
        )

    # ── Step 5: Calculate costs ───────────────────────────────────────────
    cost_breakdown = cost_calculator.calculate(
        model=routing_decision.selected_model,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
    )
    token_usage = cost_calculator.build_token_usage(
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
    )

    # ── Step 6: Compute total pipeline latency ────────────────────────────
    # This includes classification + routing + LLM call
    # Not just the LLM latency
    total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

    # ── Step 7: Log everything to PostgreSQL ──────────────────────────────
    try:
        await crud.log_request(
            db,
            request_id=request_id,
            caller_id=request.caller_id,
            prompt=request.prompt,
            difficulty_tags=[t.value for t in classification.tags],
            classifier_confidence=classification.confidence,
            routed_model=routing_decision.selected_model.value,
            routing_reason=routing_decision.reason,
            response_text=llm_response.content,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            total_tokens=llm_response.total_tokens,
            actual_cost_usd=cost_breakdown.actual_total_cost,
            baseline_gpt4o_cost_usd=cost_breakdown.baseline_total_cost,
            cost_saved_usd=cost_breakdown.cost_saved,
            latency_ms=total_latency_ms,
        )
    except Exception as e:
        # Logging failure must NEVER affect the user response
        # We log the error and continue — observability is
        # best-effort, user experience is not
        logger.error(
            "Failed to log request to DB",
            request_id=str(request_id),
            error=str(e),
        )

    logger.info(
        "Request complete",
        request_id=str(request_id),
        model=routing_decision.selected_model.value,
        total_latency_ms=f"{total_latency_ms:.1f}ms",
        cost_saved=str(cost_breakdown.cost_saved),
    )

    # ── Step 8: Return response ───────────────────────────────────────────
    return CompletionResponse(
        request_id=request_id,
        response=llm_response.content,
        routed_model=routing_decision.selected_model,
        difficulty_tags=classification.tags,
        classifier_confidence=classification.confidence,
        token_usage=token_usage,
        cost_breakdown=cost_breakdown.to_pydantic(),
        latency_ms=total_latency_ms,
    )
