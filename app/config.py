"""
Central configuration — all environment variables live here.
Every other module imports from this file, never from os.environ directly.
This makes testing trivial: override Settings, done.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Database ---
    database_url: str
    groq_api_key: str = ""
    # --- API Keys ---
    phi3_model_id: str = "llama-3.1-8b-instant"
    llama3_model_id: str = "llama-3.3-70b-versatile"
    gpt4o_model_id: str = "mixtral-8x7b-32768"

    # --- Model Pricing (per 1M tokens, USD) ---
    phi3_input_cost_per_1m: float = 0.50
    phi3_output_cost_per_1m: float = 0.50
    llama3_input_cost_per_1m: float = 0.90
    llama3_output_cost_per_1m: float = 0.90
    gpt4o_input_cost_per_1m: float = 5.00
    gpt4o_output_cost_per_1m: float = 15.00

    # --- Routing Thresholds ---
    high_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.50

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Cached singleton — Settings object is created once per process.
    lru_cache means we never re-read .env on every request.
    """
    return Settings()