from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionConfig(BaseModel):
    """Validated runtime configuration for the CLI."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    model_specs: list[str] = Field(
        ...,
        min_length=1,
        description=(
            'Model locations. Can be file paths (.pkl/.pickle/.joblib) '
            "or import strings like 'package.module:object_name'."
        ),
    )
    features: list[float] = Field(
        ...,
        min_length=1,
        description='Numeric feature vector for a single prediction.',
    )
    strategy: Literal['mean', 'vote', 'first'] = Field(
        default='mean',
        description='How to combine predictions from multiple agents.',
    )
    output_format: Literal['text', 'json'] = Field(
        default='text',
        description='How results should be printed.',
    )

    @field_validator('model_specs')
    @classmethod
    def validate_model_specs(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError('At least one model spec must be provided.')
        return cleaned

    @field_validator('features')
    @classmethod
    def validate_features(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError('At least one feature value must be provided.')
        return value

    def existing_model_paths(self) -> list[Path]:
        """Return only specs that exist on disk as paths."""
        paths: list[Path] = []
        for spec in self.model_specs:
            path = Path(spec)
            if path.exists():
                paths.append(path)
        return paths
