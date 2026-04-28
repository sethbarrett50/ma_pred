from __future__ import annotations

import logging
import sys

from pathlib import Path

DEFAULT_LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def parse_log_level(level: str | int) -> int:
    """Convert a string or int log level into a logging module constant."""
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    value = getattr(logging, normalized, None)

    if not isinstance(value, int):
        raise ValueError(f'Invalid log level: {level}')

    return value


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module or component name."""
    return logging.getLogger(name)


def setup_logging(
    level: str | int = 'INFO',
    log_file: str | Path | None = None,
    *,
    console: bool = True,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """
    Configure root logging for the application.

    Args:
        level: Logging level such as 'DEBUG', 'INFO', or a logging constant.
        log_file: Optional file path for file logging.
        console: Whether to enable console logging.
        log_format: Log message format string.
        date_format: Datetime format string.
    """
    resolved_level = parse_log_level(level)
    root_logger = logging.getLogger()

    root_logger.setLevel(resolved_level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_exception(
    logger: logging.Logger,
    message: str,
    *,
    exc: Exception | None = None,
) -> None:
    """Log an exception with traceback information."""
    if exc is not None:
        logger.exception('%s: %s', message, exc)
        return

    logger.exception(message)
