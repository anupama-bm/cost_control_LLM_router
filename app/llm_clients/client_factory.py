"""
Client Factory — maps ModelName enum to the right client instance.

"""

from functools import lru_cache
from app.schemas.pydantic_models import ModelName
from app.llm_clients.base import BaseLLMClient
from app.llm_clients.phi_client import Phi3Client
from app.llm_clients.llama_client import LLaMA3Client
from app.llm_clients.mixtral_client import MixtralClient
from app.llm_clients.base import LLMClientError


# ── Singleton clients ─────────────────────────────────────────────────────
# Each client is instantiated once and reused.
# Why? AsyncGroq internally manages a connection pool.
# Creating a new client per request would destroy and rebuild
# that connection pool on every request — catastrophic for performance.

@lru_cache(maxsize=1)
def _get_phi3_client() -> Phi3Client:
    return Phi3Client()


@lru_cache(maxsize=1)
def _get_llama3_client() -> LLaMA3Client:
    return LLaMA3Client()


@lru_cache(maxsize=1)
def _get_mixtral_client() -> MixtralClient:
    return MixtralClient()


def get_client(model: ModelName) -> BaseLLMClient:
    """
    Returns the singleton client for the given model tier.

    Usage:
        client = get_client(ModelName.LLAMA3_70B)
        response = await client.complete_with_timing(prompt)
    """
    client_map = {
        ModelName.PHI3: _get_phi3_client(),
        ModelName.LLAMA3_70B: _get_llama3_client(),
        ModelName.GPT4O: _get_mixtral_client(),
    }

    client = client_map.get(model)
    if client is None:
        raise LLMClientError(
            message=f"No client registered for model: {model}",
            model=str(model),
        )

    return client
