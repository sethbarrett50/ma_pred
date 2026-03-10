from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import TYPE_CHECKING, Any

from ma_pred.utils.model_loader import ModelLoader, SupportsPredict

if TYPE_CHECKING:
    from ma_pred.utils.config import PredictionConfig


class MultiAgentPredictor:
    """Loads multiple models and combines their outputs."""

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.models: dict[str, SupportsPredict] = ModelLoader.load_many(config.model_specs)

    def predict(self) -> dict[str, Any]:
        """Run all models on one feature vector and aggregate the result."""
        batched_features = [self.config.features]
        raw_predictions: dict[str, Any] = {}

        for model_name, model in self.models.items():
            prediction = model.predict(batched_features)
            raw_predictions[model_name] = self._extract_scalar_prediction(prediction)

        combined = self._aggregate(list(raw_predictions.values()))

        return {
            'strategy': self.config.strategy,
            'features': self.config.features,
            'predictions': raw_predictions,
            'combined_prediction': combined,
        }

    def _aggregate(self, values: list[Any]) -> Any:
        if not values:
            raise ValueError('No predictions were produced.')

        if self.config.strategy == 'first':
            return values[0]

        if self.config.strategy == 'mean':
            numeric_values = [float(value) for value in values]
            return mean(numeric_values)

        if self.config.strategy == 'vote':
            counts = Counter(values)
            return counts.most_common(1)[0][0]

        raise ValueError(f'Unknown aggregation strategy: {self.config.strategy}')

    @staticmethod
    def _extract_scalar_prediction(prediction: Any) -> Any:
        """
        Convert common model outputs into a scalar for one sample.

        Handles outputs like:
            - [0]
            - [0.73]
            - scalar values
        """
        if isinstance(prediction, (list, tuple)):
            if not prediction:
                raise ValueError('Model returned an empty prediction.')
            return prediction[0]

        return prediction
