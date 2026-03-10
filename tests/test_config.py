from ma_pred.utils.config import PredictionConfig


def test_prediction_config_valid() -> None:
    config = PredictionConfig(
        model_specs=['models/example.pkl'],
        features=[1.0, 2.0, 3.0],
        strategy='mean',
        output_format='json',
    )

    assert config.model_specs == ['models/example.pkl']
    assert config.features == [1.0, 2.0, 3.0]
    assert config.strategy == 'mean'
    assert config.output_format == 'json'
