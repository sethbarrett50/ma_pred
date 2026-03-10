from __future__ import annotations

from typing import TYPE_CHECKING

import joblib

from ma_pred.utils.model_loader import ModelLoader

if TYPE_CHECKING:
    from pathlib import Path


class DummyModel:
    def predict(self, X: list[list[float]]) -> list[float]:
        return [sum(X[0])]


def test_load_model_from_joblib_file(tmp_path: Path) -> None:
    model_path = tmp_path / 'dummy_model.joblib'
    joblib.dump(DummyModel(), model_path)

    model = ModelLoader.load(str(model_path))
    prediction = model.predict([[1.0, 2.0, 3.0]])

    assert prediction == [6.0]
