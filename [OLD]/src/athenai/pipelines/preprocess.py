"""
Preprocessing pipeline orchestrator.

Runs all preprocessing steps in order:
1. BuildAlgosPanelStep
2. BuildAlgosMetaStep
3. BuildGoodUniverseStep
4. BuildFeaturesStep
5. PreprocessBenchmarkStep

Each step is timed, validated, and logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from athenai.core.artifacts import ArtifactStore
from athenai.core.logging import get_logger, set_run_context, setup_logging
from athenai.core.monitoring import StepMonitor, timed_step
from athenai.data.base import PipelineStep
from athenai.data.algos_panel import BuildAlgosPanelStep
from athenai.data.algos_meta import BuildAlgosMetaStep, BuildGoodUniverseStep
from athenai.data.algos_features import BuildFeaturesStep
from athenai.data.benchmark import PreprocessBenchmarkStep

if TYPE_CHECKING:
    from athenai.core.config import PreprocessConfig


class PreprocessPipeline:
    """
    Orchestrates the full preprocessing pipeline.
    
    Steps run in order:
    1. build_algos_panel - Raw CSVs to panel parquet
    2. build_algos_meta - Per-algo statistics
    3. build_good_universe - Filter to quality algos
    4. build_features - Rolling features
    5. preprocess_benchmark - Benchmark data
    
    Each step:
    - Checks if outputs exist (skip if not overwrite)
    - Runs with timing
    - Validates outputs
    - Logs progress
    
    Usage:
        cfg = PreprocessConfig(root_dir=Path("data/raw/datos_competicion"))
        pipeline = PreprocessPipeline(cfg)
        results = pipeline.run(overwrite=False)
    """
    
    def __init__(self, cfg: "PreprocessConfig"):
        self.cfg = cfg
        self.logger = get_logger()
        
        # Define steps in order
        self.steps: list[PipelineStep] = [
            BuildAlgosPanelStep(),
            BuildAlgosMetaStep(),
            BuildGoodUniverseStep(),
            BuildFeaturesStep(),
            PreprocessBenchmarkStep(),
        ]
    
    def run(
        self,
        overwrite: bool = False,
        validate: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Path]:
        """
        Run the full preprocessing pipeline.
        
        Args:
            overwrite: Whether to overwrite existing outputs
            validate: Whether to run validation after each step
            run_id: Optional run ID (auto-generated if None)
        
        Returns:
            Dict mapping artifact names to paths
        """
        # Setup
        store = ArtifactStore(
            root=self.cfg.cache_dir,
            config=self.cfg,
            run_id=run_id,
        )
        store.ensure_dirs()
        
        set_run_context(run_id=store.run_id)
        setup_logging(log_file=store.run_dir / "pipeline.log")
        
        self.logger.info(f"Starting preprocessing pipeline")
        self.logger.info(f"Run ID: {store.run_id}")
        self.logger.info(f"Output dir: {store.run_dir}")
        
        monitor = StepMonitor()
        all_outputs: dict[str, Path] = {}
        
        # Run each step
        for step in self.steps:
            with monitor.track(step.name) as profile:
                try:
                    set_run_context(step_name=step.name)
                    
                    # Run step
                    outputs = step.run(store, self.cfg, overwrite=overwrite)
                    all_outputs.update(outputs)
                    
                    # Track output rows
                    for name, path in outputs.items():
                        info = store.manifest.outputs.get(name)
                        if info:
                            profile.output_rows += info.row_count
                    
                    # Validate
                    if validate:
                        step.validate(store)
                    
                except Exception as e:
                    store.add_error(f"Step {step.name} failed: {e}")
                    self.logger.error(f"Step {step.name} failed: {e}")
                    raise
        
        # Record timings
        for timing in monitor.get_timings():
            store.add_timing(timing)
        
        # Save manifest
        manifest_path = store.save_manifest()
        self.logger.info(f"Saved manifest: {manifest_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print(store.summary())
        print("=" * 60)
        print(f"\n{monitor.summary()}")
        
        return all_outputs
    
    def run_step(
        self,
        step_name: str,
        overwrite: bool = False,
        validate: bool = True,
    ) -> dict[str, Path]:
        """
        Run a single step by name.
        
        Args:
            step_name: Name of the step to run
            overwrite: Whether to overwrite
            validate: Whether to validate
        
        Returns:
            Dict of outputs from this step
        """
        step = next((s for s in self.steps if s.name == step_name), None)
        if step is None:
            available = [s.name for s in self.steps]
            raise ValueError(f"Unknown step: {step_name}. Available: {available}")
        
        store = ArtifactStore(root=self.cfg.cache_dir, config=self.cfg)
        store.ensure_dirs()
        
        set_run_context(run_id=store.run_id, step_name=step_name)
        
        with timed_step(step_name):
            outputs = step.run(store, self.cfg, overwrite=overwrite)
            if validate:
                step.validate(store)
        
        return outputs


def run_preprocessing(
    cfg: "PreprocessConfig",
    overwrite: bool = False,
    validate: bool = True,
) -> dict[str, Path]:
    """
    Convenience function to run the full preprocessing pipeline.
    
    Args:
        cfg: Configuration
        overwrite: Whether to overwrite existing outputs
        validate: Whether to validate
    
    Returns:
        Dict mapping artifact names to paths
    """
    pipeline = PreprocessPipeline(cfg)
    return pipeline.run(overwrite=overwrite, validate=validate)
