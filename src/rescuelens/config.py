"""Runtime configuration, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-5-5")
    # auto: API if ANTHROPIC_API_KEY is set, else the local Claude Code CLI (subscription).
    llm_backend: Literal["auto", "api", "claude-code"] = Field(
        default="auto", alias="RESCUELENS_LLM_BACKEND"
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="RESCUELENS_DATA_DIR")
    runs_dir: Path = Field(default=PROJECT_ROOT / "runs", alias="RESCUELENS_RUNS_DIR")
    results_dir: Path = Field(default=PROJECT_ROOT / "results", alias="RESCUELENS_RESULTS_DIR")
    prompts_dir: Path = Field(default=PROJECT_ROOT / "prompts")
    weights: Path | None = Field(default=None, alias="RESCUELENS_WEIGHTS")

    llm_max_concurrency: int = Field(default=4, ge=1, le=32, alias="RESCUELENS_LLM_CONCURRENCY")
    llm_max_cost_usd: float = Field(default=10.0, gt=0, alias="RESCUELENS_MAX_COST_USD")

    log_level: str = Field(default="INFO", alias="RESCUELENS_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="RESCUELENS_LOG_JSON")

    @property
    def default_weights(self) -> Path:
        """Explicit RESCUELENS_WEIGHTS, else the HERIDAL fine-tune if present, else COCO."""
        if self.weights is not None:
            return self.weights
        tuned = PROJECT_ROOT / "weights" / "rescuelens-yolo26s-heridal.pt"
        return tuned if tuned.exists() else PROJECT_ROOT / "weights" / "yolo26s.pt"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
