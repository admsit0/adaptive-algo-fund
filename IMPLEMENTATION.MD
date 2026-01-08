# Implementation Plan - Week 1 Baseline Strategy

## Goal

Implement a simple baseline strategy that ranks algorithms by momentum and volatility, satisfying the competition's initial requirements and constraints (10 min execution limit).

## User Review Required

IMPORTANT

The strategy relies on defining "active" algorithms. We will assume algorithms with data up to the most recent timestamp in the dataset are "active".

## Proposed Changes

### Structure

Create a `src/` directory to organize the code.

### Data Analysis & Reverse Engineering

#### [NEW]

src/benchmark_analysis.py

* Analyze

  data/trades_benchmark.csv to calculate:

  * Latent AUM (Assets Under Management).
  * Maximum Drawdown.
  * Strategy type (Mean Reversion vs Trend Following).
* Justification: Required by Phase 1 of instructions.

### Data Loading & Processing

#### [NEW]

src/data_loader.py

* `load_market_data()`: Efficiently load data for algorithms.
* get_availability_mask(): Identify active algorithms (those with recent data) to mitigate survivorship bias.
* Optimization: Use `os.scandir` or `glob` to quickly filter files.
* **[Implemented]** Use binary reading (`open(rb)`) to read only the last few bytes for availability check and metrics calculation, avoiding full CSV parsing. This reduces execution time significantly.

### Strategy Implementation

#### [NEW]

src/strategy_baseline.py

* calculate_metrics(data): Calculate 3-month Momentum and Volatility.
* rank_algorithms(metrics): Select top 20 algorithms based on:
* High Momentum.
* Low Volatility.

### Main Execution

#### [NEW]

src/main.py

* Orchestrate the pipeline:
  1. Run Benchmark Analysis (print stats).
  2. Build Availability Mask.
  3. Calculate Metrics for Active Algos.
  4. Rank and Select Top 20.
  5. Output the selection.

## Verification Plan

### Automated verification

* **Benchmark Analysis** : Run `python src/benchmark_analysis.py` and verify it prints reasonable AUM and Max Drawdown figures.
* **Strategy Output** : Run `python src/main.py` and verify it produces a list of 20 algorithms.

### Manual Verification

* **Performance** : Measure total execution time of

  src/main.py using `Measure-Command { python src/main.py }` (PowerShell) to ensure it is under 10 minutes.
