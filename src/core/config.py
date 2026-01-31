import os
from pathlib import Path

# Rutas Base
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ALGOS_DIR = DATA_DIR / "raw_algos"
BENCHMARK_DIR = DATA_DIR / "benchmark"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_LOG_DIR = PROJECT_ROOT / "training_logs"

# Crear directorios si no existen
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del Modelo
REBALANCE_FREQ = 1   # Rebalancear clusters cada día

# Archivos procesados
UNIVERSE_FILE = PROCESSED_DIR / "universe_returns.parquet"
ACTIVE_MASK_FILE = PROCESSED_DIR / "active_mask.parquet"
CLUSTERS_FILE = PROCESSED_DIR / "cluster_returns.parquet"
LABELS_FILE = PROCESSED_DIR / "cluster_labels.parquet" # Mapa: Algo -> Cluster
MACRO_FILE = PROCESSED_DIR / "macro_features.parquet"

WINDOW_SIZE = 30
N_CLUSTERS = 10

# RIESGO DURO (Aquí es donde configuras el SL y TP)
STOP_LOSS_PCT = -0.2  # Si la cartera cae un 5% en un día, VENDEMOS TODO (Cierre forzoso)
TAKE_PROFIT_PCT = 0.3 # Si ganamos un 20% en un día, CERRAMOS para asegurar (Opcional)

# Costes (Para que no haga trading a lo loco)
TRADING_COST_BPS = 0.0 # 5 basis points por cambio

TRAIN_START_DATE = "2020-06-01"
TRAIN_END_DATE   = "2024-12-29"

TEST_START_DATE  = "2024-12-30"
TEST_END_DATE    = "2024-12-31"

# Modo Cross-Validation: Si True, ignora las fechas fijas y hace bucles anuales
ENABLE_CROSS_VAL = False
