"""
Tier 3 — Strongest reasoning model.
Used for: high_context_analysis, low-confidence escalations
Groq model: mixtral-8x7b-32768

Why Mixtral for Tier 3?
    Mixture of Experts architecture — 8 expert networks, each
    prompt token is routed to the 2 most relevant experts.
    This gives GPT-3.5-level reasoning with faster inference.
    The 32768 context window handles long documents that would
    overflow smaller models.

    Key differentiator from LLaMA3:
        LLaMA3 70B = one large dense network
        Mixtral 8x7B = 8 specialized networks working together
        Different architecture = genuinely different capability profile
        Mixtral excels at: structured reasoning, long context, instruction following
"""

from app.llm_clients.groq_base import GroqBaseLLMClient
from app.config import get_settings

settings = get_settings()


class MixtralClient(GroqBaseLLMClient):
    """
    Tier 3 LLM Client.
    Inherits all Groq API logic from GroqBaseLLMClient.
    Only responsibility: declare which model to use.
    """

    @property
    def model_name(self) -> str:
        return settings.gpt4o_model_id  # mixtral-8x7b-32768