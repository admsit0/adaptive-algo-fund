import pandas as pd
from src.core.config import UNIVERSE_FILE, ACTIVE_MASK_FILE
from src.features.macro_proxy import build_macro_features
from src.features.clustering import run_rolling_clustering

if __name__ == "__main__":
    print("Cargando universo...")
    returns = pd.read_parquet(UNIVERSE_FILE)
    mask = pd.read_parquet(ACTIVE_MASK_FILE)
    
    build_macro_features(returns)
    run_rolling_clustering(returns, mask)
    