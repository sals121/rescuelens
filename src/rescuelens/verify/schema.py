"""Structured output contract between RescueLens and Claude."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Posture = Literal["standing", "lying", "sitting", "unknown"]
Priority = Literal["high", "medium", "low"]


class Verification(BaseModel):
    """Claude's verdict on a single detection candidate."""

    model_config = ConfigDict(extra="forbid")

    is_person: bool = Field(description="True if the highlighted object is a human being.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in is_person, 0-1.")
    evidence: str = Field(
        max_length=400,
        description="One or two sentences citing the visual cues behind the verdict.",
    )
    posture: Posture = Field(description="Body posture if a person, otherwise 'unknown'.")
    priority: Priority = Field(
        description="Rescue priority: high = possibly injured/immobile person, "
        "medium = person needing a check, low = unlikely to be a person.",
    )
