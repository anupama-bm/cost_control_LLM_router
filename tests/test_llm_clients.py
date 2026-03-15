"""
Test the client layer end to end.
These tests make REAL Groq API calls — run sparingly.
Usage: python -m pytest tests/test_llm_clients.py -v -s
"""

import pytest
from app.llm_clients.client_factory import get_client
from app.schemas.pydantic_models import ModelName


@pytest.mark.asyncio
async def test_phi3_client_responds():
    client = get_client(ModelName.PHI3)
    response = await client.complete_with_timing(
        prompt="What is 2 + 2? Answer in one word.",
        max_tokens=10,
    )
    print(f"\nModel: {response.model_id}")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms:.1f}ms")
    assert len(response.content) > 0
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_llama3_client_responds():
    client = get_client(ModelName.LLAMA3_70B)
    response = await client.complete_with_timing(
        prompt="Write a Python function that reverses a string.",
        max_tokens=200,
    )
    print(f"\nModel: {response.model_id}")
    print(f"Response: {response.content[:100]}...")
    print(f"Latency: {response.latency_ms:.1f}ms")
    assert "def" in response.content  # Should contain Python code
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_mixtral_client_responds():
    client = get_client(ModelName.GPT4O)
    response = await client.complete_with_timing(
        prompt="Explain the tradeoffs between SQL and NoSQL databases in 3 sentences.",
        max_tokens=300,
    )
    print(f"\nModel: {response.model_id}")
    print(f"Response: {response.content[:150]}...")
    print(f"Latency: {response.latency_ms:.1f}ms")
    assert len(response.content) > 50
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_factory_returns_correct_clients():
    phi3 = get_client(ModelName.PHI3)
    llama3 = get_client(ModelName.LLAMA3_70B)
    mixtral = get_client(ModelName.GPT4O)

    assert phi3.model_name == "llama-3.1-8b-instant"
    assert llama3.model_name == "llama-3.3-70b-versatile"
    assert mixtral.model_name == "mixtral-8x7b-32768"


def test_singleton_behavior():
    """
    Same client instance returned every time.
    Critical for connection pool reuse.
    """
    client1 = get_client(ModelName.PHI3)
    client2 = get_client(ModelName.PHI3)
    assert client1 is client2  # Same object in memory