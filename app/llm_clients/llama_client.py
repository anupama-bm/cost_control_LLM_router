"""
Tier 2 — Balanced model.
Used for: code_generation, multi_step_reasoning
Groq model: llama-3.3-70b-versatile

Why this for Tier 2?
    70B parameters gives strong reasoning and code quality.
    LLaMA 3.3 is Meta's most capable open model as of late 2024.
    On Groq's LPU it runs faster than GPT-4o despite being larger.
    Cost is ~10x less than GPT-4o for similar quality on
    medium-complexity tasks.
"""

from app.llm_clients.groq_base import GroqBaseLLMClient
from app.config import get_settings

settings = get_settings()


class LLaMA3Client(GroqBaseLLMClient):
    """
    Tier 2 LLM Client.
    Inherits all Groq API logic from GroqBaseLLMClient.
    Only responsibility: declare which model to use.
    """

    @property
    def model_name(self) -> str:
        return settings.llama3_model_id  # llama-3.3-70b-versatile