"""
Adaptive Routing Engine — the learning system.

Responsibility:
    After each feedback submission, analyze whether the routing
    rule for that tag+model combination needs to change.

Design principles:
    - Minimum sample size before adjusting (avoid noise)
    - Thresholds are configurable, not hardcoded
    - Every adjustment is logged to DB with full reasoning
    - The engine never adjusts the same tag twice in quick
      succession (cooldown period prevents thrashing)
    - All decisions are explainable — no black box

Interview talking point:
    "The adaptive engine uses statistical thresholds, not ML.
    This was a deliberate choice. Statistical rules are:
    auditable (you can explain every decision),
    debuggable (you can trace why a rule changed),
    predictable (no model drift or retraining needed).
    ML would be appropriate at much higher scale where
    the pattern space is too complex for explicit rules."
"""

import structlog
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.router.model_router import (
    BASE_ROUTING_TABLE,
    escalate_model,
    demote_model,
    MODEL_TIERS,
)
from app.schemas.pydantic_models import DifficultyTag, ModelName

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLDS
# These control how sensitive the system is to feedback.
#
# UNDERPOWERED_THRESHOLD = 0.30
#   If 30%+ of feedback says "too weak" → escalate
#   Reasoning: 3 in 10 users unhappy is significant
#
# OVERKILL_THRESHOLD = 0.40
#   If 40%+ of feedback says "overkill" → demote
#   Reasoning: We set this higher than underpowered because
#   quality problems are worse than cost inefficiency.
#   We'd rather slightly overspend than deliver poor quality.
#
# MIN_SAMPLE_SIZE = 10
#   Need at least 10 feedback entries before adjusting.
#   Reasoning: 3/5 underpowered (60%) could be random noise.
#   3/10 underpowered (30%) starts to be a real signal.
#
# COOLDOWN_HOURS = 24
#   Don't adjust the same tag more than once per 24 hours.
#   Reasoning: Prevents thrashing — rapid back-and-forth
#   between tiers due to conflicting feedback in short bursts.
# ─────────────────────────────────────────────────────────

UNDERPOWERED_THRESHOLD = 0.30
OVERKILL_THRESHOLD = 0.40
MIN_SAMPLE_SIZE = 10
COOLDOWN_HOURS = 24


class AdaptiveRoutingEngine:
    """
    Stateless engine — all state lives in PostgreSQL.

    Why stateless?
        In a multi-instance deployment (multiple API servers),
        all instances share the same PostgreSQL. If state lived
        in memory, each instance would have different routing rules.
        By keeping all state in DB, all instances always agree.

    Interview talking point:
        "Stateless services are horizontally scalable by definition.
        Any instance can handle any request because they all read
        from the same source of truth — PostgreSQL. This is the
        12-factor app principle: store state in a backing service,
        not in application memory."
    """

    async def analyze_and_adjust(
        self,
        db: AsyncSession,
        tag: str,
        model: str,
    ) -> bool:
        """
        Core method — called after every feedback submission.

        Analyzes feedback statistics for the given tag+model
        combination and adjusts routing rules if thresholds
        are crossed.

        Args:
            db:    Database session
            tag:   The difficulty tag of the original request
                   e.g. "code_generation"
            model: The model that handled the request
                   e.g. "llama-3-70b"

        Returns:
            True if a routing adjustment was made
            False if no adjustment was needed

        Flow:
            1. Get feedback stats for this tag+model
            2. Check minimum sample size
            3. Check cooldown period
            4. Check underpowered threshold
            5. Check overkill threshold
            6. Apply adjustment if needed
        """

        logger.info(
            "Analyzing feedback for adaptive routing",
            tag=tag,
            model=model,
        )

        # ── Step 1: Get feedback statistics ───────────────────────────────
        stats = await crud.get_feedback_stats_for_tag(
            db,
            tag=tag,
            model=model,
            lookback_days=7,
        )

        logger.info(
            "Feedback stats retrieved",
            tag=tag,
            model=model,
            total_feedback=stats["total_feedback"],
            underpowered_rate=stats["underpowered_rate"],
            overkill_rate=stats["overkill_rate"],
            avg_rating=stats["avg_rating"],
        )

        # ── Step 2: Check minimum sample size ─────────────────────────────
        # Too few samples = statistical noise, not signal.
        # Don't adjust routing based on 2 or 3 data points.
        if stats["total_feedback"] < MIN_SAMPLE_SIZE:
            logger.info(
                "Insufficient feedback for adjustment",
                tag=tag,
                total_feedback=stats["total_feedback"],
                minimum_required=MIN_SAMPLE_SIZE,
            )
            return False

        # ── Step 3: Check cooldown period ─────────────────────────────────
        # Prevents rapid thrashing between tiers
        in_cooldown = await self._check_cooldown(db, tag)
        if in_cooldown:
            logger.info(
                "Tag in cooldown period, skipping adjustment",
                tag=tag,
                cooldown_hours=COOLDOWN_HOURS,
            )
            return False

        # ── Step 4: Convert model string to ModelName enum ────────────────
        try:
            current_model = ModelName(model)
        except ValueError:
            logger.warning(
                "Unknown model name, cannot adjust routing",
                model=model,
            )
            return False

        # ── Step 5: Check underpowered threshold ──────────────────────────
        # Too many people saying quality was poor →
        # this model is not strong enough for this tag
        if stats["underpowered_rate"] >= UNDERPOWERED_THRESHOLD:
            return await self._apply_escalation(
                db=db,
                tag=tag,
                current_model=current_model,
                stats=stats,
            )

        # ── Step 6: Check overkill threshold ──────────────────────────────
        # Too many people saying this was overkill →
        # we're wasting money on an overpowered model
        if stats["overkill_rate"] >= OVERKILL_THRESHOLD:
            return await self._apply_demotion(
                db=db,
                tag=tag,
                current_model=current_model,
                stats=stats,
            )

        logger.info(
            "No routing adjustment needed",
            tag=tag,
            model=model,
            underpowered_rate=stats["underpowered_rate"],
            overkill_rate=stats["overkill_rate"],
        )
        return False

    async def _apply_escalation(
        self,
        db: AsyncSession,
        tag: str,
        current_model: ModelName,
        stats: dict,
    ) -> bool:
        """
        Escalates the routing tier for a tag.
        Called when underpowered_rate exceeds threshold.

        Example:
            code_generation was routing to PHI3
            30%+ feedback said underpowered
            → Now routes to LLAMA3_70B
        """
        new_model, was_escalated = escalate_model(current_model)

        if not was_escalated:
            # Already at the strongest model — can't escalate further
            logger.warning(
                "Cannot escalate — already at maximum tier",
                tag=tag,
                model=current_model.value,
            )
            return False

        reason = (
            f"Underpowered rate {stats['underpowered_rate']:.1%} exceeded "
            f"threshold {UNDERPOWERED_THRESHOLD:.1%} over {MIN_SAMPLE_SIZE}+ "
            f"feedback samples. Average rating: {stats['avg_rating']:.1f}/5. "
            f"Escalating from {current_model.value} to {new_model.value}."
        )

        await crud.save_routing_adjustment(
            db,
            difficulty_tag=tag,
            previous_model=current_model.value,
            new_model=new_model.value,
            reason=reason,
            trigger_feedback_count=stats["total_feedback"],
            underpowered_rate=stats["underpowered_rate"],
            overkill_rate=stats["overkill_rate"],
        )

        logger.info(
            "ROUTING ESCALATION APPLIED",
            tag=tag,
            from_model=current_model.value,
            to_model=new_model.value,
            underpowered_rate=stats["underpowered_rate"],
            avg_rating=stats["avg_rating"],
        )

        return True

    async def _apply_demotion(
        self,
        db: AsyncSession,
        tag: str,
        current_model: ModelName,
        stats: dict,
    ) -> bool:
        """
        Demotes the routing tier for a tag.
        Called when overkill_rate exceeds threshold.

        Example:
            simple_summarization was routing to LLAMA3_70B
            40%+ feedback said overkill
            → Now routes to PHI3 (cheaper, still sufficient)
        """
        new_model, was_demoted = demote_model(current_model)

        if not was_demoted:
            # Already at the cheapest model — can't demote further
            logger.warning(
                "Cannot demote — already at minimum tier",
                tag=tag,
                model=current_model.value,
            )
            return False

        reason = (
            f"Overkill rate {stats['overkill_rate']:.1%} exceeded "
            f"threshold {OVERKILL_THRESHOLD:.1%} over {MIN_SAMPLE_SIZE}+ "
            f"feedback samples. Average rating: {stats['avg_rating']:.1f}/5. "
            f"Demoting from {current_model.value} to {new_model.value}."
        )

        await crud.save_routing_adjustment(
            db,
            difficulty_tag=tag,
            previous_model=current_model.value,
            new_model=new_model.value,
            reason=reason,
            trigger_feedback_count=stats["total_feedback"],
            underpowered_rate=stats["underpowered_rate"],
            overkill_rate=stats["overkill_rate"],
        )

        logger.info(
            "ROUTING DEMOTION APPLIED",
            tag=tag,
            from_model=current_model.value,
            to_model=new_model.value,
            overkill_rate=stats["overkill_rate"],
            avg_rating=stats["avg_rating"],
        )

        return True

    async def _check_cooldown(
        self,
        db: AsyncSession,
        tag: str,
    ) -> bool:
        """
        Checks if this tag was adjusted recently.
        Returns True if still in cooldown (skip adjustment).
        Returns False if cooldown has passed (can adjust).

        Why cooldown matters:
            Without it, 10 rapid feedback submissions could
            trigger escalation, then 10 more trigger demotion,
            cycling back and forth. Cooldown forces the system
            to settle after each adjustment before re-evaluating.
        """
        from sqlalchemy import select, desc
        from app.db.models import RoutingAdjustment

        # Find the most recent adjustment for this tag
        result = await db.execute(
            select(RoutingAdjustment)
            .where(RoutingAdjustment.difficulty_tag == tag)
            .order_by(desc(RoutingAdjustment.timestamp))
            .limit(1)
        )
        last_adjustment = result.scalar_one_or_none()

        if last_adjustment is None:
            return False  # Never been adjusted — no cooldown

        cooldown_cutoff = datetime.utcnow() - timedelta(hours=COOLDOWN_HOURS)
        still_in_cooldown = last_adjustment.timestamp > cooldown_cutoff

        if still_in_cooldown:
            hours_remaining = (
                last_adjustment.timestamp - cooldown_cutoff
            ).seconds // 3600
            logger.info(
                "Cooldown active",
                tag=tag,
                hours_remaining=hours_remaining,
            )

        return still_in_cooldown