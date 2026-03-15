"""
Model Router — the decision engine of the routing platform.

Responsibility:
    Take classifier output + feedback history → select the right model.

Three layers of logic:
    1. Base routing table   (static, tag → model)
    2. Confidence fallback  (dynamic, low confidence = escalate)
    3. Adaptive routing     (learned, feedback history adjusts tiers)

"""

import structlog
from typing import Dict
from app.schemas.pydantic_models import (
    ClassificationResult,
    DifficultyTag,
    ModelName,
    RoutingDecision,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────
# BASE ROUTING TABLE
# The default mapping from difficulty tag to model tier.
# This is the "static knowledge" layer.
# Changing this table changes the entire routing behavior —
# one place, full control.
# ─────────────────────────────────────────────────────────

BASE_ROUTING_TABLE: Dict[DifficultyTag, ModelName] = {
    DifficultyTag.SIMPLE_SUMMARIZATION:   ModelName.PHI3,
    DifficultyTag.INFORMATION_EXTRACTION: ModelName.PHI3,
    DifficultyTag.CODE_GENERATION:        ModelName.LLAMA3_70B,
    DifficultyTag.MULTI_STEP_REASONING:   ModelName.LLAMA3_70B,
    DifficultyTag.HIGH_CONTEXT_ANALYSIS:  ModelName.GPT4O,
}

# ─────────────────────────────────────────────────────────
# MODEL TIER ORDERING
# Used for escalation and demotion logic.
# Higher index = stronger model.
# ─────────────────────────────────────────────────────────

MODEL_TIERS = [
    ModelName.PHI3,        # tier 0 — cheapest
    ModelName.LLAMA3_70B,  # tier 1 — medium
    ModelName.GPT4O,       # tier 2 — strongest
]


def escalate_model(model: ModelName) -> tuple[ModelName, bool]:
    """
    Move up one tier. Returns (new_model, was_escalated).
    If already at max tier, stays there.

    >>> escalate_model(ModelName.PHI3)
    (ModelName.LLAMA3_70B, True)

    >>> escalate_model(ModelName.GPT4O)
    (ModelName.GPT4O, False)   ← already at max, no escalation
    """
    current_index = MODEL_TIERS.index(model)
    if current_index < len(MODEL_TIERS) - 1:
        return MODEL_TIERS[current_index + 1], True
    return model, False


def demote_model(model: ModelName) -> tuple[ModelName, bool]:
    """
    Move down one tier. Returns (new_model, was_demoted).
    If already at min tier, stays there.

    >>> demote_model(ModelName.GPT4O)
    (ModelName.LLAMA3_70B, True)

    >>> demote_model(ModelName.PHI3)
    (ModelName.PHI3, False)    ← already at min, no demotion
    """
    current_index = MODEL_TIERS.index(model)
    if current_index > 0:
        return MODEL_TIERS[current_index - 1], True
    return model, False


class ModelRouter:
    """
    Stateless router — routing rules are passed in, not stored here.
    This makes the router fully testable without a database.

    The adaptive_overrides dict is loaded fresh per request from
    the database (via the CRUD layer). This means routing changes
    from feedback take effect on the NEXT request after feedback
    is processed — no restart required.

    Why stateless?
        If we stored rules inside the router object, we'd need to
        invalidate them when feedback arrives. Stateless = always
        fresh, no cache invalidation problem.
    """

    def route(
        self,
        classification: ClassificationResult,
        primary_tag: DifficultyTag,
        adaptive_overrides: Dict[str, str] = None,
    ) -> RoutingDecision:
        """
        Main routing method. Applies all three layers in sequence.

        Args:
            classification:    Output from PromptClassifier.classify()
            primary_tag:       Output from PromptClassifier.get_primary_tag()
            adaptive_overrides: Dict of {tag_value: model_name} from DB feedback
                                e.g. {"code_generation": "gpt-4o"}

        Returns:
            RoutingDecision with selected model and human-readable reason
        """

        if adaptive_overrides is None:
            adaptive_overrides = {}

        logger.info(
            "Routing request",
            primary_tag=primary_tag.value,
            confidence=classification.confidence,
            adaptive_overrides=adaptive_overrides,
        )

        # ── Layer 1: Adaptive Override ────────────────────────────────────
        # Check if feedback history has changed the tier for this tag.
        # This is the "learned knowledge" layer — takes highest priority.
        if primary_tag.value in adaptive_overrides:
            overridden_model = ModelName(adaptive_overrides[primary_tag.value])
            reason = (
                f"Adaptive routing override applied for '{primary_tag.value}': "
                f"feedback history escalated this tag to {overridden_model.value}"
            )
            logger.info("Adaptive override applied", model=overridden_model.value)
            return RoutingDecision(selected_model=overridden_model, reason=reason)

        # ── Layer 2: Base Routing Table ───────────────────────────────────
        # Look up the default model for this tag.
        base_model = BASE_ROUTING_TABLE.get(primary_tag, ModelName.GPT4O)
        reason = f"Base routing: '{primary_tag.value}' maps to {base_model.value}"

        # ── Layer 3: Confidence-Based Adjustment ──────────────────────────
        # Low confidence = classifier wasn't sure = escalate to be safe.
        # High confidence = classifier is certain = trust the base tier.

        if classification.confidence < settings.low_confidence_threshold:
            # Confidence below 0.5 — classifier is not sure what this prompt is.
            # Safe move: escalate one tier up.
            escalated_model, was_escalated = escalate_model(base_model)

            if was_escalated:
                reason = (
                    f"Low confidence ({classification.confidence:.2f}) on "
                    f"'{primary_tag.value}' — escalated from "
                    f"{base_model.value} to {escalated_model.value} for safety"
                )
                logger.info(
                    "Escalated due to low confidence",
                    from_model=base_model.value,
                    to_model=escalated_model.value,
                    confidence=classification.confidence,
                )
                return RoutingDecision(
                    selected_model=escalated_model,
                    reason=reason
                )
            else:
                reason = (
                    f"Low confidence ({classification.confidence:.2f}) but "
                    f"already at strongest model {base_model.value}"
                )

        elif classification.confidence >= settings.high_confidence_threshold:
            # Confidence above 0.8 — classifier is very sure.
            # Trust the base routing table completely.
            reason = (
                f"High confidence ({classification.confidence:.2f}) on "
                f"'{primary_tag.value}' — using base model {base_model.value}"
            )
            logger.info(
                "High confidence routing",
                model=base_model.value,
                confidence=classification.confidence,
            )

        else:
            # Confidence between 0.5 and 0.8 — moderate confidence.
            # Trust the base routing table but log it.
            reason = (
                f"Moderate confidence ({classification.confidence:.2f}) on "
                f"'{primary_tag.value}' — using base model {base_model.value}"
            )

        return RoutingDecision(selected_model=base_model, reason=reason)


# ─────────────────────────────────────────────────────────
# ROUTING SUMMARY — for logging and transparency
# ─────────────────────────────────────────────────────────

def explain_routing_table() -> str:
    """
    Returns a human-readable explanation of current routing rules.
    Useful for debugging and admin endpoints.

    """
    lines = ["Current Base Routing Table:"]
    lines.append("─" * 50)
    for tag, model in BASE_ROUTING_TABLE.items():
        lines.append(f"  {tag.value:<30} → {model.value}")
    lines.append("─" * 50)
    lines.append(f"  Low confidence  (< {settings.low_confidence_threshold}) → escalate one tier")
    lines.append(f"  High confidence (≥ {settings.high_confidence_threshold}) → trust base tier")
    return "\n".join(lines)
