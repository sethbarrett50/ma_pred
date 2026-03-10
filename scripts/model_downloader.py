from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelName = Literal['tinyllama', 'llama32_1b', 'deepseek_coder_1_3b']

MODEL_SPECS: dict[ModelName, dict[str, str]] = {
    'tinyllama': {
        'repo_id': 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        'filename': 'tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf',
    },
    'llama32_1b': {
        'repo_id': 'bartowski/Llama-3.2-1B-Instruct-GGUF',
        'filename': 'Llama-3.2-1B-Instruct-Q5_K_M.gguf',
    },
    'deepseek_coder_1_3b': {
        'repo_id': 'TheBloke/deepseek-coder-1.3b-instruct-GGUF',
        'filename': 'deepseek-coder-1.3b-instruct.Q5_K_M.gguf',
    },
}


class DownloadModelsConfig(BaseModel):
    """Validated configuration for model downloads."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    models: list[ModelName] = Field(
        ...,
        min_length=1,
        description='Named models to download.',
    )
    output_dir: Path = Field(
        default=Path('models'),
        description='Directory where models will be saved.',
    )

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        return value.expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Download starter GGUF models into ./models.',
    )
    parser.add_argument(
        '--model',
        dest='models',
        action='append',
        choices=sorted(MODEL_SPECS.keys()),
        help='Model to download. May be repeated.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all predefined models.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models'),
        help='Directory where models should be stored.',
    )
    return parser


def resolve_requested_models(
    requested_models: list[str] | None,
    download_all: bool,
) -> list[ModelName]:
    """Resolve final model list from CLI arguments."""
    if download_all:
        return list(MODEL_SPECS.keys())

    if not requested_models:
        raise ValueError('Provide --all or at least one --model.')

    valid_models = set(MODEL_SPECS.keys())
    invalid = [model for model in requested_models if model not in valid_models]
    if invalid:
        raise ValueError(f'Invalid model(s): {", ".join(invalid)}')

    return [cast('ModelName', model) for model in requested_models]


def download_model(repo_id: str, filename: str, output_dir: Path) -> None:
    """Download a model file from Hugging Face using uvx."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'uvx',
        'hf',
        'download',
        repo_id,
        filename,
        '--local-dir',
        str(output_dir),
    ]

    print(f'Downloading {filename} from {repo_id} into {output_dir}')
    subprocess.run(cmd, check=True)


def run(config: DownloadModelsConfig) -> None:
    """Download all models listed in the validated config."""
    for model_name in config.models:
        spec = MODEL_SPECS[model_name]
        download_model(
            repo_id=spec['repo_id'],
            filename=spec['filename'],
            output_dir=config.output_dir,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        selected_models = resolve_requested_models(
            requested_models=args.models,
            download_all=args.all,
        )
        config = DownloadModelsConfig(
            models=selected_models,
            output_dir=args.output_dir,
        )
        run(config)
    except subprocess.CalledProcessError as exc:
        print(
            f'Download command failed with exit code {exc.returncode}',
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
