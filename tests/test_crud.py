"""
CRUD tests — these require a real PostgreSQL connection.
Set up a test database before running.

Usage: python -m pytest tests/test_crud.py -v -s
"""

import pytest
import uuid
from decimal import Decimal
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import Base
from app.db import crud

# Use a separate test database — never test against production
TEST_DATABASE_URL = "postgresql+asyncpg://llmrouter:password@localhost:5432/llm_router_test"


@pytest.fixture(scope="session")
async def test_db():
    """
    Creates fresh test database schema for the test session.
    Drops all tables after tests complete.
    """
    engine = create_async_engine(TEST_DATABASE_URL)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    # Teardown — clean slate for next test run
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_log_and_retrieve_request(test_db):
    request_id = uuid.uuid4()

    logged = await crud.log_request(
        test_db,
        request_id=request_id,
        caller_id="test-service",
        prompt="What is the capital of France?",
        difficulty_tags=["simple_summarization"],
        classifier_confidence=0.95,
        routed_model="phi-3",
        routing_reason="High confidence simple task",
        response_text="The capital of France is Paris.",
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        actual_cost_usd=Decimal("0.000015"),
        baseline_gpt4o_cost_usd=Decimal("0.000250"),
        cost_saved_usd=Decimal("0.000235"),
        latency_ms=245.5,
    )
    await test_db.commit()

    # Retrieve and verify
    fetched = await crud.get_request_by_id(test_db, request_id)
    assert fetched is not None
    assert fetched.prompt == "What is the capital of France?"
    assert fetched.routed_model == "phi-3"
    assert float(fetched.cost_saved_usd) == pytest.approx(0.000235)

    print(f"\nLogged request: {fetched.id}")
    print(f"Cost saved: ${fetched.cost_saved_usd}")


@pytest.mark.asyncio
async def test_save_and_retrieve_feedback(test_db):
    # First create a request to attach feedback to
    request_id = uuid.uuid4()
    await crud.log_request(
        test_db,
        request_id=request_id,
        caller_id=None,
        prompt="Explain recursion.",
        difficulty_tags=["multi_step_reasoning"],
        classifier_confidence=0.72,
        routed_model="llama-3-70b",
        routing_reason="Medium complexity",
        response_text="Recursion is...",
        input_tokens=15,
        output_tokens=150,
        total_tokens=165,
        actual_cost_usd=Decimal("0.000149"),
        baseline_gpt4o_cost_usd=Decimal("0.002325"),
        cost_saved_usd=Decimal("0.002176"),
        latency_ms=1200.0,
    )
    await test_db.commit()

    feedback = await crud.save_feedback(
        test_db,
        request_id=request_id,
        rating=2,
        underpowered=True,
        overkill=False,
        comment="Response was too shallow for this question",
    )
    await test_db.commit()

    assert feedback.rating == 2
    assert feedback.underpowered is True
    print(f"\nFeedback saved: {feedback.id}")
    print(f"Rating: {feedback.rating}/5")
    print(f"Underpowered flag: {feedback.underpowered}")


@pytest.mark.asyncio
async def test_cost_summary(test_db):
    summary = await crud.get_cost_summary(test_db, lookback_days=30)
    print(f"\nCost Summary:")
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Total saved:    ${summary['total_saved_usd']}")
    print(f"  Savings %:      {summary['savings_percentage']}%")
    assert "total_requests" in summary
    assert "savings_percentage" in summary




