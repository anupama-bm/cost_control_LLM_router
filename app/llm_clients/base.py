"""
Abstract Base Client — the contract every LLM client must fulfill.

Why ABC (Abstract Base Class)?
    Python won't let you instantiate a class that has unimplemented
    abstract methods. This means if someone adds a new LLM client
    and forgets to implement .complete(), they get an error at import
    time — not at 3am when a request fails in production.

Interview talking point:
    "We used the Abstract Base Class pattern to enforce interface
    consistency across all LLM providers. This is the same pattern
    used in Python's own standard library — asyncio, collections.abc.
    It gives us compile-time safety in a dynamic language."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized response object returned by every LLM client.

    No matter which provider we called, the rest of the system
    always gets back this same structure. This is the adapter pattern
    — we adapt provider-specific responses into our internal format.

    Interview talking point:
        "Every provider returns slightly different response shapes.
        OpenAI, Groq, Anthropic — all different. By normalizing into
        LLMResponse at the client boundary, the rest of the system
        is completely provider-agnostic. Switching providers is
        contained to one file."
    """
    content: str  # The actual text response
    input_tokens: int  # Tokens consumed by the prompt
    output_tokens: int  # Tokens generated in the response
    total_tokens: int  # input + output
    model_id: str  # Exact model that responded (for audit trail)
    latency_ms: float  # How long the API call took


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM provider clients.

    Every subclass MUST implement:
        - complete(prompt, max_tokens, temperature) → LLMResponse
        - model_name property → str

    Every subclass GETS for free:
        - complete_with_timing() → wraps complete() with latency tracking
        - Structured logging
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The model identifier string sent to the provider API.
        e.g. "llama-3.1-8b-instant", "mixtral-8x7b-32768"
        """
        pass

    @abstractmethod
    async def complete(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a prompt to the LLM and return a normalized response.

        Args:
            prompt:       The full prompt text to send
            max_tokens:   Maximum tokens to generate in response
            temperature:  0.0 = deterministic, 1.0 = creative

        Returns:
            LLMResponse with content, token counts, and latency

        Raises:
            LLMClientError: on API failure after retries
        """
        pass

    async def complete_with_timing(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Wraps complete() with precise latency measurement.

        Why not measure inside complete()?
            Because complete() is implemented by subclasses.
            We want timing to be consistent regardless of how
            a subclass implements its API call. Measuring here,
            in the base class, guarantees consistency.

        This is the Template Method pattern — the base class
        defines the algorithm structure, subclasses fill in steps.
        """
        logger.info(
            "LLM call starting",
            model=self.model_name,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
        )

        start_time = time.perf_counter()  # High-precision timer

        try:
            response = await self.complete(prompt, max_tokens, temperature)

            # Inject timing into response
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            response.latency_ms = elapsed_ms

            logger.info(
                "LLM call complete",
                model=self.model_name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=f"{elapsed_ms:.1f}ms",
            )

            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM call failed",
                model=self.model_name,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=f"{elapsed_ms:.1f}ms",
            )
            raise


class LLMClientError(Exception):
    """
    Raised when an LLM client fails after all retries.

    Wrapping provider exceptions in our own exception type means:
    - Route handlers catch LLMClientError, not GroqError or OpenAIError
    - Switching providers doesn't change exception handling upstream
    - We can attach structured context (model, prompt_length, etc.)
    """

    def __init__(self, message: str, model: str, original_error: Exception = None):
        super().__init__(message)
        self.model = model
        self.original_error = original_error