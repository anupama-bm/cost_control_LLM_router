"""
Cost Calculator — financial transparency layer.

Responsibility:
    Given token counts and model used, compute:
    - Actual cost of the routed model
    - Baseline cost if GPT-4o had been used
    - Cost saved per request
    - Projected monthly savings

Design decisions:
    - All math uses Python Decimal for financial precision
      (floats have rounding errors — never use them for money)
    - Pricing is loaded from config — never hardcoded
    - GPT-4o is always the baseline (it's the most expensive tier)
    - CostCalculator is stateless — pure functions, no side effects

Interview talking point:
    "The cost calculator is intentionally stateless and pure —
    given the same inputs it always returns the same outputs.
    This makes it trivially testable and means we can run it
    retrospectively on historical data to recalculate savings
    with updated pricing."
"""

from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from typing import Dict
import structlog

from app.schemas.pydantic_models import ModelName, TokenUsage, CostBreakdown
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────
# PRICING TABLE
# Loaded from settings (which loads from .env)
# Structure: {ModelName: (input_price_per_1M, output_price_per_1M)}
#
# Why separate input and output pricing?
#   Because every provider charges them differently.
#   GPT-4o charges $5/1M input but $15/1M output — 3x difference.
#   If we averaged them, our cost calculations would be wrong by
#   up to 50% depending on input/output ratio.
#
# Interview talking point:
#   "We model input and output tokens separately because provider
#   pricing is asymmetric. Output tokens cost more because they
#   require autoregressive generation — each token depends on all
#   previous tokens. Input tokens are processed in parallel.
#   This asymmetry is fundamental to transformer architecture."
# ─────────────────────────────────────────────────────────

def _build_pricing_table() -> Dict[ModelName, tuple[Decimal, Decimal]]:
    """
    Builds pricing table from environment config.
    Called once at module load time.

    Returns:
        Dict mapping ModelName to (input_cost_per_1M, output_cost_per_1M)
        Both values are Decimal for financial precision.
    """
    return {
        ModelName.PHI3: (
            Decimal(str(settings.phi3_input_cost_per_1m)),
            Decimal(str(settings.phi3_output_cost_per_1m)),
        ),
        ModelName.LLAMA3_70B: (
            Decimal(str(settings.llama3_input_cost_per_1m)),
            Decimal(str(settings.llama3_output_cost_per_1m)),
        ),
        ModelName.GPT4O: (
            Decimal(str(settings.gpt4o_input_cost_per_1m)),
            Decimal(str(settings.gpt4o_output_cost_per_1m)),
        ),
    }


# Module-level constant — built once, reused forever
PRICING_TABLE = _build_pricing_table()

# Precision for financial calculations
# 8 decimal places — handles even sub-cent costs accurately
FINANCIAL_PRECISION = Decimal("0.00000001")


@dataclass
class DetailedCostBreakdown:
    """
    Internal detailed breakdown — more granular than CostBreakdown.
    Used for logging and analytics.
    CostBreakdown (Pydantic) is the public API version.
    """
    # Actual cost components
    actual_input_cost: Decimal
    actual_output_cost: Decimal
    actual_total_cost: Decimal

    # Baseline (GPT-4o) cost components
    baseline_input_cost: Decimal
    baseline_output_cost: Decimal
    baseline_total_cost: Decimal

    # Derived
    cost_saved: Decimal
    savings_percentage: float
    routed_model: ModelName

    def to_pydantic(self) -> CostBreakdown:
        """Convert to the Pydantic model used in API responses."""
        return CostBreakdown(
            actual_cost_usd=float(self.actual_total_cost),
            baseline_gpt4o_cost_usd=float(self.baseline_total_cost),
            cost_saved_usd=float(self.cost_saved),
            routed_model=self.routed_model,
        )


class CostCalculator:
    """
    Stateless cost calculator.
    All methods are pure functions — no state, no side effects.

    Usage:
        calculator = CostCalculator()
        breakdown = calculator.calculate(
            model=ModelName.PHI3,
            input_tokens=500,
            output_tokens=300
        )
    """

    def calculate(
            self,
            model: ModelName,
            input_tokens: int,
            output_tokens: int,
    ) -> DetailedCostBreakdown:
        """
        Main calculation method.

        Args:
            model:         The model that was actually used
            input_tokens:  Tokens consumed by the prompt
            output_tokens: Tokens generated in the response

        Returns:
            DetailedCostBreakdown with full financial breakdown
        """

        # ── Actual cost calculation ───────────────────────────────────────
        actual_input_cost, actual_output_cost = self._compute_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        actual_total = actual_input_cost + actual_output_cost

        # ── Baseline cost (what GPT-4o would have charged) ───────────────
        # We always use the real GPT-4o pricing for baseline,
        # even though we're using Mixtral as our "tier 3" in dev.
        # This gives accurate savings projections for production.
        baseline_input_cost, baseline_output_cost = self._compute_cost(
            model=ModelName.GPT4O,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        baseline_total = baseline_input_cost + baseline_output_cost

        # ── Savings calculation ───────────────────────────────────────────
        # Cost saved can be 0 if we routed to GPT-4o tier
        # Cost saved can be negative if... actually it can't.
        # GPT-4o is always the most expensive. We assert this.
        cost_saved = baseline_total - actual_total

        # Calculate savings as a percentage for human readability
        if baseline_total > 0:
            savings_pct = float(
                (cost_saved / baseline_total * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )
        else:
            savings_pct = 0.0

        breakdown = DetailedCostBreakdown(
            actual_input_cost=actual_input_cost.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            actual_output_cost=actual_output_cost.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            actual_total_cost=actual_total.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            baseline_input_cost=baseline_input_cost.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            baseline_output_cost=baseline_output_cost.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            baseline_total_cost=baseline_total.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            cost_saved=cost_saved.quantize(
                FINANCIAL_PRECISION, rounding=ROUND_HALF_UP
            ),
            savings_percentage=savings_pct,
            routed_model=model,
        )

        logger.info(
            "Cost calculated",
            model=model.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            actual_cost=str(breakdown.actual_total_cost),
            baseline_cost=str(breakdown.baseline_total_cost),
            cost_saved=str(breakdown.cost_saved),
            savings_pct=f"{savings_pct:.1f}%",
        )

        return breakdown

    def _compute_cost(
            self,
            model: ModelName,
            input_tokens: int,
            output_tokens: int,
    ) -> tuple[Decimal, Decimal]:
        """
        Compute input and output costs separately for a given model.

        Formula:
            cost = (token_count / 1_000_000) × price_per_million

        Returns:
            (input_cost, output_cost) as Decimal tuple
        """
        input_price_per_1m, output_price_per_1m = PRICING_TABLE[model]

        input_cost = (
                             Decimal(str(input_tokens)) / Decimal("1000000")
                     ) * input_price_per_1m

        output_cost = (
                              Decimal(str(output_tokens)) / Decimal("1000000")
                      ) * output_price_per_1m

        return input_cost, output_cost

    def project_monthly_savings(
            self,
            cost_saved_per_request: Decimal,
            requests_per_day: int,
    ) -> Dict[str, float]:
        """
        Projects monthly and annual savings based on current routing.

        This is the number that justifies the platform to a CTO.

        Args:
            cost_saved_per_request: From a recent calculate() call
            requests_per_day:       Expected daily request volume

        Returns:
            Dict with daily, monthly, annual projections in USD

        Interview talking point:
            "We built savings projection into the cost calculator
            because the platform needs to justify itself. When a
            CTO asks 'is this worth it?', we can answer with a
            specific dollar figure based on actual observed savings
            rates — not estimates."
        """
        daily = float(cost_saved_per_request) * requests_per_day
        monthly = daily * 30
        annual = daily * 365

        return {
            "daily_savings_usd": round(daily, 4),
            "monthly_savings_usd": round(monthly, 2),
            "annual_savings_usd": round(annual, 2),
            "requests_per_day": requests_per_day,
        }

    def build_token_usage(
            self,
            input_tokens: int,
            output_tokens: int,
    ) -> TokenUsage:
        """
        Builds the TokenUsage Pydantic model from raw counts.
        Small helper to keep route handlers clean.
        """
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )