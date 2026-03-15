"""
Cost calculator tests — pure math, no API calls, instant execution.
These should all pass in under 1 second total.

Usage: python -m pytest tests/test_cost_calculator.py -v -s
"""

import pytest
from decimal import Decimal
from app.cost.cost_calculator import CostCalculator, PRICING_TABLE
from app.schemas.pydantic_models import ModelName

calculator = CostCalculator()


def test_phi3_cheaper_than_gpt4o():
    """
    Core invariant: PHI3 must always cost less than GPT-4O.
    If this fails, the pricing table is misconfigured.
    """
    phi3_breakdown = calculator.calculate(
        model=ModelName.PHI3,
        input_tokens=1000,
        output_tokens=500,
    )
    gpt4o_breakdown = calculator.calculate(
        model=ModelName.GPT4O,
        input_tokens=1000,
        output_tokens=500,
    )
    assert phi3_breakdown.actual_total_cost < gpt4o_breakdown.actual_total_cost


def test_cost_saved_is_positive_for_cheaper_models():
    """
    Routing to PHI3 or LLAMA3 should always save money vs GPT-4o.
    """
    for model in [ModelName.PHI3, ModelName.LLAMA3_70B]:
        breakdown = calculator.calculate(
            model=model,
            input_tokens=1000,
            output_tokens=500,
        )
        assert breakdown.cost_saved >= Decimal("0"), \
            f"Cost saved should be >= 0 for {model}, got {breakdown.cost_saved}"


def test_cost_saved_is_zero_for_gpt4o():
    """
    Routing to GPT-4o saves nothing — it IS the baseline.
    """
    breakdown = calculator.calculate(
        model=ModelName.GPT4O,
        input_tokens=1000,
        output_tokens=500,
    )
    assert breakdown.cost_saved == Decimal("0")


def test_exact_cost_calculation():
    """
    Manually verify the math is correct.

    PHI3 pricing: $0.50/1M input, $0.50/1M output
    500 input tokens:  (500/1_000_000) * 0.50 = $0.00025
    300 output tokens: (300/1_000_000) * 0.50 = $0.00015
    Total actual:                                $0.00040

    GPT4O pricing: $5.00/1M input, $15.00/1M output
    500 input tokens:  (500/1_000_000) * 5.00  = $0.00250
    300 output tokens: (300/1_000_000) * 15.00 = $0.00450
    Total baseline:                              $0.00700

    Cost saved: $0.00700 - $0.00040 = $0.00660
    """
    breakdown = calculator.calculate(
        model=ModelName.PHI3,
        input_tokens=500,
        output_tokens=300,
    )

    assert breakdown.actual_total_cost == Decimal("0.00040000")
    assert breakdown.baseline_total_cost == Decimal("0.00700000")
    assert breakdown.cost_saved == Decimal("0.00660000")

    print(f"\nActual cost:   ${breakdown.actual_total_cost}")
    print(f"Baseline cost: ${breakdown.baseline_total_cost}")
    print(f"Cost saved:    ${breakdown.cost_saved}")
    print(f"Savings:       {breakdown.savings_percentage}%")


def test_monthly_savings_projection():
    """
    100,000 requests/day, each saving $0.0066 = $660/day = $19,800/month
    """
    projection = calculator.project_monthly_savings(
        cost_saved_per_request=Decimal("0.0066"),
        requests_per_day=100_000,
    )
    print(f"\nDaily savings:   ${projection['daily_savings_usd']}")
    print(f"Monthly savings: ${projection['monthly_savings_usd']}")
    print(f"Annual savings:  ${projection['annual_savings_usd']}")

    assert projection["monthly_savings_usd"] == 19800.0
    assert projection["annual_savings_usd"] == 240900.0


def test_token_usage_builder():
    usage = calculator.build_token_usage(
        input_tokens=500,
        output_tokens=300,
    )
    assert usage.input_tokens == 500
    assert usage.output_tokens == 300
    assert usage.total_tokens == 800


def test_pydantic_conversion():
    """
    Verify DetailedCostBreakdown converts to Pydantic CostBreakdown correctly.
    """
    breakdown = calculator.calculate(
        model=ModelName.PHI3,
        input_tokens=500,
        output_tokens=300,
    )
    pydantic_model = breakdown.to_pydantic()
    assert pydantic_model.actual_cost_usd == float(breakdown.actual_total_cost)
    assert pydantic_model.cost_saved_usd == float(breakdown.cost_saved)
    assert pydantic_model.routed_model == ModelName.PHI3


