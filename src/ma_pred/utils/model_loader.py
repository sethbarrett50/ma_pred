from __future__ import annotations

import importlib
import pickle

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import joblib

from llama_cpp import Llama


@runtime_checkable
class SupportsPredict(Protocol):
    """Protocol for simple prediction-capable models."""

    def predict(self, X: list[list[float]]) -> Any:
        """Run prediction on batched inputs."""
        ...


class GGUFModelWrapper:
    """Adapter that wraps a GGUF LLM with a simple predict interface."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    def predict(self, X: list[list[float]]) -> list[str]:
        """
        Convert numeric inputs into prompts and return text outputs.

        This preserves compatibility with the repo's current predictor
        interface, which expects a predict(...) method.
        """
        outputs: list[str] = []

        for row in X:
            prompt = self._build_prompt(row)
            text = self._run_prompt(prompt)
            outputs.append(text)

        return outputs

    def _run_prompt(self, prompt: str) -> str:
        """
        Run a single non-streaming completion and return generated text.
        """
        response = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.2,
            stream=False,
        )

        if isinstance(response, Iterator):
            raise TypeError(
                "Expected non-streaming completion response, "
                "but got a streaming iterator."
            )

        response_dict = cast("dict[str, Any]", response)
        choices = response_dict.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response did not contain any choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("LLM response choice had an unexpected shape.")

        text = first_choice.get("text")
        if not isinstance(text, str):
            raise ValueError("LLM response did not contain text output.")

        return text.strip()

    @staticmethod
    def _build_prompt(features: list[float]) -> str:
        feature_text = ", ".join(str(value) for value in features)
        return (
            "You are a prediction agent.\n"
            f"Input features: [{feature_text}]\n"
            "Provide a short prediction or classification."
        )


class ModelLoader:
    """Utility class for loading models from files or import strings."""

    SUPPORTED_FILE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".gguf"}

    @classmethod
    def load(cls, spec: str) -> SupportsPredict:
        """
        Load a model from a file path or import string.

        Supported examples:
            - models/my_model.pkl
            - models/my_model.joblib
            - models/my_model.gguf
            - my_package.models:my_model
        """
        path = Path(spec)
        if path.exists():
            return cls._load_from_path(path)

        if ":" in spec:
            return cls._load_from_import(spec)

        raise ValueError(
            f"Unsupported model spec '{spec}'. "
            "Use an existing file path or an import string like "
            "'package.module:object_name'."
        )

    @classmethod
    def load_many(cls, specs: list[str]) -> dict[str, SupportsPredict]:
        """Load multiple models keyed by their original spec."""
        return {spec: cls.load(spec) for spec in specs}

    @classmethod
    def _load_from_path(cls, path: Path) -> SupportsPredict:
        suffix = path.suffix.lower()

        if suffix not in cls.SUPPORTED_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported model file type '{path.suffix}'. "
                f"Supported: {sorted(cls.SUPPORTED_FILE_SUFFIXES)}"
            )

        model: Any
        if suffix == ".joblib":
            model = joblib.load(path)
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as file_handle:
                model = pickle.load(file_handle)
        elif suffix == ".gguf":
            model = GGUFModelWrapper(path)
        else:
            raise ValueError(f"Unsupported model type: {path.suffix}")

        cls._validate_model(model, source=str(path))
        return model

    @classmethod
    def _load_from_import(cls, spec: str) -> SupportsPredict:
        module_name, object_name = spec.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        model = getattr(module, object_name)
        cls._validate_model(model, source=spec)
        return model

    @staticmethod
    def _validate_model(model: Any, source: str) -> None:
        if not isinstance(model, SupportsPredict):
            raise TypeError(
                f"Loaded object from '{source}' does not provide "
                "a compatible .predict(...) method."
            )
