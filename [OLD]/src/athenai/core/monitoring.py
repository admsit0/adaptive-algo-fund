"""
Monitoring utilities for AthenAI pipeline.

Provides:
- Timing decorators
- Memory tracking
- Row count tracking
- Step profiling
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generator

from athenai.core.artifacts import StepTiming
from athenai.core.logging import get_logger, log_step_start, log_step_end


@dataclass
class StepProfile:
    """Profile information for a single step execution."""
    step_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    input_rows: int = 0
    output_rows: int = 0
    memory_mb: float = 0.0
    notes: list[str] = field(default_factory=list)
    
    def finish(self) -> None:
        """Mark the step as finished."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_timing(self) -> StepTiming:
        """Convert to StepTiming for manifest."""
        return StepTiming(
            step_name=self.step_name,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat() if self.end_time else "",
            duration_seconds=self.duration_seconds,
        )


@contextmanager
def timed_step(step_name: str) -> Generator[StepProfile, None, None]:
    """
    Context manager for timing a pipeline step.
    
    Usage:
        with timed_step("build_panel") as profile:
            # ... do work ...
            profile.output_rows = len(df)
        print(f"Took {profile.duration_seconds:.2f}s")
    """
    profile = StepProfile(step_name=step_name)
    log_step_start(step_name)
    
    try:
        yield profile
    finally:
        profile.finish()
        log_step_end(
            step_name,
            rows=profile.output_rows if profile.output_rows > 0 else None,
            duration=profile.duration_seconds,
        )


def timed(func: Callable) -> Callable:
    """
    Decorator to time a function and log its duration.
    
    Usage:
        @timed
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger()
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
    
    return wrapper


class StepMonitor:
    """
    Monitor for tracking pipeline step execution.
    
    Collects:
    - Timing per step
    - Row counts
    - Warnings
    
    Usage:
        monitor = StepMonitor()
        with monitor.track("step1"):
            ...
        print(monitor.summary())
    """
    
    def __init__(self) -> None:
        self.profiles: list[StepProfile] = []
        self._current: StepProfile | None = None
    
    @contextmanager
    def track(self, step_name: str) -> Generator[StepProfile, None, None]:
        """Track a step execution."""
        profile = StepProfile(step_name=step_name)
        self._current = profile
        log_step_start(step_name)
        
        try:
            yield profile
        finally:
            profile.finish()
            self.profiles.append(profile)
            self._current = None
            log_step_end(
                step_name,
                rows=profile.output_rows if profile.output_rows > 0 else None,
                duration=profile.duration_seconds,
            )
    
    def current(self) -> StepProfile | None:
        """Get the currently executing profile."""
        return self._current
    
    def total_duration(self) -> float:
        """Total duration of all tracked steps."""
        return sum(p.duration_seconds for p in self.profiles)
    
    def get_timings(self) -> list[StepTiming]:
        """Get all step timings for manifest."""
        return [p.to_timing() for p in self.profiles]
    
    def summary(self) -> str:
        """Get a summary of all tracked steps."""
        if not self.profiles:
            return "No steps tracked."
        
        lines = ["Step Execution Summary:", "-" * 50]
        for p in self.profiles:
            row_info = f", {p.output_rows:,} rows" if p.output_rows > 0 else ""
            lines.append(f"  {p.step_name}: {p.duration_seconds:.2f}s{row_info}")
        
        lines.append("-" * 50)
        lines.append(f"Total: {self.total_duration():.2f}s")
        
        return "\n".join(lines)


def estimate_memory_mb() -> float:
    """
    Estimate current process memory usage in MB.
    
    Note: This is a best-effort estimate and may not be available on all platforms.
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0
