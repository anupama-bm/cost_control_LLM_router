"""
Prompt Classifier — the intelligence layer of the routing platform.

Responsibility: Take a raw prompt, return difficulty tags + confidence score.

Design decisions worth knowing for interviews:
- We use an LLM (not regex/rules) because natural language is ambiguous
- We force JSON output via system prompt — more reliable than parsing free text
- We validate the JSON with Pydantic — never trust LLM output blindly
- Confidence score drives fallback logic in the router
- The classifier uses the FASTEST/CHEAPEST model — it's overhead, minimize it
- Tenacity handles retries — LLMs occasionally return malformed JSON
"""

import json
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import AsyncGroq
from app.config import get_settings
from app.schemas.pydantic_models import ClassificationResult, DifficultyTag

# Structured logger — outputs JSON logs in production, readable in dev
logger = structlog.get_logger(__name__)

settings = get_settings()


# ─────────────────────────────────────────────────────────
# THE SYSTEM PROMPT — this is the most critical string
# in the entire classifier. Every word matters.
# ─────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """
You are a prompt difficulty classifier for an LLM routing system.

Your job: analyze the user's prompt and return a JSON object — nothing else.
No explanation. No markdown. No code blocks. Raw JSON only.

Classify the prompt into one or more of these difficulty tags:
- simple_summarization: condensing text, TL;DR, abstracting content
- information_extraction: pulling facts, dates, names, structured data from text
- code_generation: writing, debugging, explaining, or refactoring code
- multi_step_reasoning: math problems, logic puzzles, chain-of-thought tasks
- high_context_analysis: legal docs, research papers, complex comparisons, strategic analysis

Rules:
1. You MUST return 1-3 tags maximum. Most prompts need only 1.
2. Confidence must be between 0.0 and 1.0
3. High confidence (>0.8) = you are very sure about the classification
4. Low confidence (<0.5) = the prompt is ambiguous or crosses multiple categories
5. reasoning must be one sentence explaining your decision

Return EXACTLY this JSON structure:
{
  "tags": ["tag1", "tag2"],
  "confidence": 0.85,
  "reasoning": "One sentence explanation"
}
"""

# ─────────────────────────────────────────────────────────
# DIFFICULTY TAG WEIGHTS
# Used to pick the PRIMARY tag when multiple are returned.
# Higher weight = harder = needs a stronger model.
# This is a key interview talking point — we reduce multi-tag
# results to a single routing decision cleanly.
# ─────────────────────────────────────────────────────────

TAG_COMPLEXITY_WEIGHT = {
    DifficultyTag.SIMPLE_SUMMARIZATION: 1,
    DifficultyTag.INFORMATION_EXTRACTION: 2,
    DifficultyTag.CODE_GENERATION: 3,
    DifficultyTag.MULTI_STEP_REASONING: 4,
    DifficultyTag.HIGH_CONTEXT_ANALYSIS: 5,
}


class PromptClassifier:
    """
    Stateless classifier — no memory between calls.
    Instantiated once at app startup and reused (see main.py).

    Why stateless? So we can run multiple instances behind a
    load balancer without any shared state issues.
    """

    def __init__(self):
        # AsyncGroq client — reused across requests (connection pooling)
        self.client = AsyncGroq(api_key=settings.groq_api_key)

        # The cheapest, fastest model for classification overhead
        # llama-3.1-8b-instant is Groq's fastest model
        self.model = "llama-3.1-8b-instant"

        logger.info("PromptClassifier initialized", model=self.model)

    @retry(
        stop=stop_after_attempt(3),           # Try 3 times max
        wait=wait_exponential(min=1, max=4),  # Wait 1s, 2s, 4s between retries
        reraise=True                           # If all 3 fail, bubble up the exception
    )
    async def classify(self, prompt: str) -> ClassificationResult:
        """
        Main classification method.

        Flow:
        1. Send prompt to LLM with structured system prompt
        2. Parse JSON response
        3. Validate with Pydantic
        4. Return ClassificationResult

        The @retry decorator handles transient failures automatically.
        This is production-grade — raw LLM calls fail ~2% of the time.
        """

        logger.info(
            "Classifying prompt",
            prompt_length=len(prompt),
            model=self.model
        )

        try:
            # ── Step 1: Call the LLM ──────────────────────────────
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": CLASSIFIER_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        # We prepend a clear instruction to reinforce JSON-only output
                        "content": f"Classify this prompt:\n\n{prompt}"
                    }
                ],
                temperature=0.1,    # Low temperature = deterministic, consistent output
                max_tokens=150,     # Classification needs very few tokens
            )

            raw_content = response.choices[0].message.content.strip()

            logger.debug("Raw classifier response", raw=raw_content)

            # ── Step 2: Parse JSON ────────────────────────────────
            # LLMs sometimes wrap JSON in ```json ``` blocks despite instructions
            # This strips those out defensively
            cleaned = self._strip_markdown_fences(raw_content)
            parsed = json.loads(cleaned)

            # ── Step 3: Validate with Pydantic ───────────────────
            # If the LLM hallucinated an invalid tag, Pydantic catches it here
            result = ClassificationResult(
                tags=[DifficultyTag(tag) for tag in parsed["tags"]],
                confidence=float(parsed["confidence"]),
                reasoning=parsed.get("reasoning", "No reasoning provided")
            )

            logger.info(
                "Classification complete",
                tags=[t.value for t in result.tags],
                confidence=result.confidence,
                reasoning=result.reasoning
            )

            return result

        except json.JSONDecodeError as e:
            # JSON parse failed — log and retry via @retry decorator
            logger.warning(
                "Classifier returned invalid JSON, will retry",
                error=str(e),
                raw_content=raw_content
            )
            raise  # Tenacity catches this and retries

        except KeyError as e:
            # JSON was valid but missing expected fields
            logger.warning(
                "Classifier JSON missing expected field",
                missing_field=str(e),
                parsed=parsed
            )
            raise

        except Exception as e:
            # Unexpected error — log full context for debugging
            logger.error(
                "Unexpected classifier error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def get_primary_tag(self, result: ClassificationResult) -> DifficultyTag:
        """
        When a prompt gets multiple tags, we need ONE tag to drive routing.
        We pick the highest-complexity tag — err on the side of quality.

        Example:
            tags = [simple_summarization, code_generation]
            weights = [1, 3]
            primary = code_generation  ← correct, code needs a stronger model

        Interview talking point:
            "We resolve multi-tag ambiguity by taking the highest complexity tag.
            This is a conservative strategy — we'd rather slightly overspend than
            return a poor quality response. The feedback loop corrects overkill
            over time by learning which tag combinations are actually simple."
        """
        return max(result.tags, key=lambda tag: TAG_COMPLEXITY_WEIGHT.get(tag, 0))

    def _strip_markdown_fences(self, text: str) -> str:
        """
        Defensively remove markdown code fences from LLM output.
        LLMs are trained on markdown and sometimes can't help themselves.

        Handles:
````json { ... } ```
``` { ... } ```
            Just { ... }   ← returned unchanged
        """
        if text.startswith("```"):
            # Find the first newline after the opening fence
            first_newline = text.find("\n")
            # Find the closing fence
            last_fence = text.rfind("```")

            if first_newline != -1 and last_fence > first_newline:
                return text[first_newline:last_fence].strip()

        return text
