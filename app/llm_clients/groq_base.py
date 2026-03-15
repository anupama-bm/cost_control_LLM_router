"""
Shared Groq client base — all three of our model tiers use Groq.
This intermediate base class holds the Groq API setup so each
tier client only needs to specify its model name.

Why a separate intermediate class?
    If we ever add an OpenAI client, it would have its own
    intermediate base with OpenAI setup. The three-level hierarchy:

    BaseLLMClient          ← universal contract
        └── GroqBaseLLMClient   ← Groq-specific setup
                ├── Phi3Client       ← tier 1 model
                ├── LLaMA3Client     ← tier 2 model
                └── MixtralClient    ← tier 3 model

    This way adding GPT-4o later means:
        BaseLLMClient
            ├── GroqBaseLLMClient
            └── OpenAIBaseLLMClient  ← new file, zero changes elsewhere
                    └── GPT4oClient
"""

from groq import AsyncGroq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import structlog
from app.config import get_settings
from app.llm_clients.base import BaseLLMClient, LLMResponse, LLMClientError

logger = structlog.get_logger(__name__)
settings = get_settings()


class GroqBaseLLMClient(BaseLLMClient):
    """
    Groq-specific implementation of BaseLLMClient.
    Handles authentication, retry logic, and response normalization
    for all Groq-hosted models.
    """

    def __init__(self):
        # Single AsyncGroq client instance — connection pooled internally
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        # Only retry on transient errors, not on bad requests
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    async def complete(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.7,
    ) -> LLMResponse:
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Defined by each subclass
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # ── Normalize to our internal LLMResponse format ──────────
            # Groq returns: response.choices[0].message.content
            # OpenAI returns: same structure (intentional compatibility)
            # Our system always sees: LLMResponse.content

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_id=self.model_name,
                latency_ms=0.0,  # Filled in by complete_with_timing()
            )

        except Exception as e:
            # Wrap provider exception in our own type
            raise LLMClientError(
                message=f"Groq API call failed: {str(e)}",
                model=self.model_name,
                original_error=e,
            )
