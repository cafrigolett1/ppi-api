"""Application settings, read from the environment and validated at startup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["presidio-be", "llm-zero-shot", "llm-few-shot"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "PII Anonymisation API"
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # -- engines ----------------------------------------------------------
    default_engine: EngineName = "presidio-be"
    spacy_model: str = "en_core_web_lg"
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    cohere_api_key: SecretStr | None = None
    llm_model: str = Field(
        default="command-a-03-2025",
        description="Pinned version. A floating alias changes behaviour "
        "between deploys with no code change.",
    )

    # -- input limits -----------------------------------------------------
    max_input_tokens: int = Field(
        default=8_000,
        ge=1,
        description="Per-document ceiling tokens ",
    )
    max_batch_size: int = Field(default=50, ge=1)
    max_batch_tokens: int = Field(
        default=40_000,
        ge=1,
        description="Total across a batch.",
    )
    chars_per_token: float = Field(default=4.0, gt=0)

    # -- storage ----------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///./data/requests.db",
        description="SQLite by default so the service runs with no "
        "infrastructure. Point at Postgres for anything shared.",
    )
    store_input_text: bool = Field(
        default=True,
        description="Persist the raw request text. Convenient for debugging, "
        "but it means the database holds the exact data this service exists "
        "to remove. Turn it off in production unless you have a reason.",
    )
    store_output_text: bool = Field(
        default=True,
        description="Persist the redacted output. Safe to keep on - it is "
        "the anonymised form.",
    )

    data_dir: Path = Path("data")

    @property
    def few_shot_path(self) -> Path:
        return self.data_dir / "few_shot_examples.json"

    @property
    def llm_available(self) -> bool:
        return self.cohere_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
