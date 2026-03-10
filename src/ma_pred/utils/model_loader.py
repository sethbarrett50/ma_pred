from __future__ import annotations

import importlib
import pickle

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib


@runtime_checkable
class SupportsPredict(Protocol):
    """Protocol for simple prediction-capable models."""

    def predict(self, X: list[list[float]]) -> Any:
        """Run prediction on batched inputs."""
        ...


class ModelLoader:
    """Utility class for loading models from files or import strings."""

    SUPPORTED_FILE_SUFFIXES = {'.pkl', '.pickle', '.joblib'}

    @classmethod
    def load(cls, spec: str) -> SupportsPredict:
        """
        Load a model from a file path or import string.

        Supported examples:
            - models/my_model.pkl
            - models/my_model.joblib
            - my_package.models:my_model
        """
        path = Path(spec)
        if path.exists():
            return cls._load_from_path(path)

        if ':' in spec:
            return cls._load_from_import(spec)

        raise ValueError(
            f"Unsupported model spec '{spec}'. "
            'Use an existing file path or an import string like '
            "'package.module:object_name'."
        )

    @classmethod
    def load_many(cls, specs: list[str]) -> dict[str, SupportsPredict]:
        """Load multiple models keyed by their original spec."""
        return {spec: cls.load(spec) for spec in specs}

    @classmethod
    def _load_from_path(cls, path: Path) -> SupportsPredict:
        if path.suffix not in cls.SUPPORTED_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported model file type '{path.suffix}'. Supported: {sorted(cls.SUPPORTED_FILE_SUFFIXES)}"
            )

        if path.suffix == '.joblib':
            model = joblib.load(path)
        else:
            with path.open('rb') as file_handle:
                model = pickle.load(file_handle)

        cls._validate_model(model, source=str(path))
        return model

    @classmethod
    def _load_from_import(cls, spec: str) -> SupportsPredict:
        module_name, object_name = spec.split(':', maxsplit=1)
        module = importlib.import_module(module_name)
        model = getattr(module, object_name)
        cls._validate_model(model, source=spec)
        return model

    @staticmethod
    def _validate_model(model: Any, source: str) -> None:
        if not isinstance(model, SupportsPredict):
            raise TypeError(f"Loaded object from '{source}' does not provide a compatible .predict(...) method.")
