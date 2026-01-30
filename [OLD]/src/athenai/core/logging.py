"""
Logging utilities for AthenAI pipeline.

Provides:
- Structured logging with run context
- Rich console output (optional)
- File logging per run
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Context variables for run-level info
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_step_name: ContextVar[str | None] = ContextVar("step_name", default=None)


def set_run_context(run_id: str | None = None, step_name: str | None = None) -> None:
    """Set the current run context."""
    if run_id is not None:
        _run_id.set(run_id)
    if step_name is not None:
        _step_name.set(step_name)


def get_run_id() -> str | None:
    """Get the current run ID."""
    return _run_id.get()


def get_step_name() -> str | None:
    """Get the current step name."""
    return _step_name.get()


class ContextFilter(logging.Filter):
    """Add run context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get() or "no-run"
        record.step_name = _step_name.get() or "no-step"
        return True


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    use_rich: bool = True,
) -> None:
    """
    Setup logging for the pipeline.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to write logs to
        use_rich: Use rich console handler if available
    """
    # Root logger for athenai
    logger = logging.getLogger("athenai")
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Format
    fmt = "%(asctime)s | %(run_id)s | %(step_name)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Context filter
    context_filter = ContextFilter()
    
    # Console handler
    try:
        if use_rich:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
            # Rich has its own formatting, use simpler format
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            raise ImportError("Using standard handler")
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
    
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)


def get_logger(name: str = "athenai") -> logging.Logger:
    """
    Get a logger with the given name.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


# Simple print helpers that work without Rich
def log_info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)


def log_warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)


def log_error(msg: str) -> None:
    """Log error message."""
    get_logger().error(msg)


def log_step_start(step_name: str) -> None:
    """Log the start of a pipeline step."""
    set_run_context(step_name=step_name)
    get_logger().info(f"▶ Starting: {step_name}")


def log_step_end(step_name: str, rows: int | None = None, duration: float | None = None) -> None:
    """Log the end of a pipeline step."""
    parts = [f"✓ Completed: {step_name}"]
    if rows is not None:
        parts.append(f"rows={rows:,}")
    if duration is not None:
        parts.append(f"time={duration:.2f}s")
    get_logger().info(" | ".join(parts))
