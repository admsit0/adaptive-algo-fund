import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from .config import RAW_ALGOS_DIR, UNIVERSE_FILE, ACTIVE_MASK_FILE

def load_single_algo(filepath):
    """Carga un solo CSV y devuelve la serie de precios de cierre."""
    try:
        # Asumimos nombre archivo = ID del algo
        algo_id = os.path.basename(filepath).replace('.csv', '')
        df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
        df = df[~df.index.duplicated(keep='first')] # Eliminar duplicados
        return algo_id, df['close']
    except Exception:
        return None, None

def build_universe():
    """Lee 14k CSVs y genera matrices alineadas."""
    files = [os.path.join(RAW_ALGOS_DIR, f) for f in os.listdir(RAW_ALGOS_DIR) if f.endswith('.csv')]
    print(f"🚀 Ingestando {len(files)} algoritmos...")

    results = Parallel(n_jobs=-1)(delayed(load_single_algo)(f) for f in tqdm(files))
    
    # Construir diccionario {id: serie}
    data_dict = {res[0]: res[1] for res in results if res[0] is not None}
    
    # Crear DataFrame maestro (alinea fechas automáticamente)
    full_df = pd.DataFrame(data_dict).sort_index()
    
    # 1. Matriz de Retornos
    returns_df = full_df.pct_change().fillna(0)
    
    # 2. Máscara de Actividad (True si el algo existía y tenía precio ese día)
    # Importante para las Bases: No invertir en algos con NaN/0 constantes
    active_mask = full_df.notna() & (full_df != 0)
    
    # Guardar
    returns_df.to_parquet(UNIVERSE_FILE)
    active_mask.to_parquet(ACTIVE_MASK_FILE)
    print(f"✅ Universo guardado: {returns_df.shape}")
    