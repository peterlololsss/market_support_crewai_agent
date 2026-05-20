from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    llm_base_url: str = "https://llm.yanfuinvest.com/v1"
    llm_provider: str = "openai"
    llm_model: str = "deepseek-v4-pro"
    llm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0)
    llm_max_tokens: int = Field(default=1200, gt=0)
    crewai_verbose: bool = False
    crewai_max_iter: int = Field(default=5, gt=0)
    crewai_max_execution_time: int = Field(default=60, gt=0)


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("YANFU_LLM_BASE_URL", "https://llm.yanfuinvest.com/v1"),
        llm_provider=os.getenv("YANFU_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("YANFU_LLM_MODEL", "deepseek-v4-pro"),
        llm_api_key=os.getenv("YANFU_LLM_API_KEY") or None,
        llm_timeout_seconds=_float_env("YANFU_LLM_TIMEOUT_SECONDS", 30.0),
        llm_temperature=_float_env("YANFU_LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_int_env("YANFU_LLM_MAX_TOKENS", 1200),
        crewai_verbose=_bool_env("CREWAI_VERBOSE", False),
        crewai_max_iter=_int_env("CREWAI_MAX_ITER", 5),
        crewai_max_execution_time=_int_env("CREWAI_MAX_EXECUTION_TIME", 60),
    )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

