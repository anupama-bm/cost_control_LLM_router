"""
Tier 1 — Cheapest, fastest model.
Used for: simple_summarization, information_extraction
Groq model: llama-3.1-8b-instant

Why this for Tier 1?
    8B parameters = tiny memory footprint = ultra-low latency.
    Groq's LPU chip makes this model respond in ~100-200ms.
    For simple tasks, this is indistinguishable from GPT-4o
    in quality but costs 10x less.
"""

from app.llm_clients.groq_base import GroqBaseLLMClient
from app.config import get_settings

settings = get_settings()


class Phi3Client(GroqBaseLLMClient):
    """
    Tier 1 LLM Client.
    Inherits all Groq API logic from GroqBaseLLMClient.
    Only responsibility: declare which model to use.
    """

    @property
    def model_name(self) -> str:
        return settings.phi3_model_id  # llama-3.1-8b-instant