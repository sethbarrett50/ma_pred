from __future__ import annotations

import argparse
import json
import sys

from ma_pred.predictor import MultiAgentPredictor
from ma_pred.utils.config import PredictionConfig
from ma_pred.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='ma-pred',
        description='Run a simple multi-agent prediction workflow.',
    )
    parser.add_argument(
        '--model',
        dest='model_specs',
        action='append',
        required=True,
        help=('Model spec to load. May be repeated. Examples: models/model.pkl or package.module:object_name'),
    )
    parser.add_argument(
        '--features',
        nargs='+',
        type=float,
        required=True,
        help='Feature vector for a single sample.',
    )
    parser.add_argument(
        '--strategy',
        choices=['mean', 'vote', 'first'],
        default='mean',
        help='How to combine predictions from multiple models.',
    )
    parser.add_argument(
        '--output-format',
        choices=['text', 'json'],
        default='text',
        help='How to print results.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PredictionConfig(
        model_specs=args.model_specs,
        features=args.features,
        strategy=args.strategy,
        output_format=args.output_format,
    )

    predictor = MultiAgentPredictor(config)
    results = predictor.predict()

    if config.output_format == 'json':
        print(json.dumps(results, indent=2, default=str))
        return

    print('Multi-Agent Prediction')
    print('----------------------')
    print(f'Strategy: {results["strategy"]}')
    print(f'Features: {results["features"]}')
    print('Agent predictions:')
    for model_name, prediction in results['predictions'].items():
        print(f'  - {model_name}: {prediction}')
    print(f'Combined prediction: {results["combined_prediction"]}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
