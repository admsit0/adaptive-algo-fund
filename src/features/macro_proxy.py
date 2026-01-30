import pandas as pd
from ..core.config import MACRO_FILE

def build_macro_features(returns_df):
    """Calcula indicadores sintéticos basados en el comportamiento de la multitud."""
    print("🧠 Calculando Macro Sintética...")
    
    # 1. Sentimiento de Mercado (Retorno promedio del universo)
    mkt_ret = returns_df.mean(axis=1)
    
    # 2. Índice de Miedo (Dispersión estándar transversal)
    # Si los algos discrepan mucho (alta std), hay caos/incertidumbre.
    mkt_dispersion = returns_df.std(axis=1)
    
    # 3. Breadth (Amplitud): % de algos positivos hoy
    mkt_breadth = (returns_df > 0).mean(axis=1)
    
    macro_df = pd.concat([mkt_ret, mkt_dispersion, mkt_breadth], axis=1)
    macro_df.columns = ['syn_mkt_ret', 'syn_vix', 'syn_breadth']
    
    macro_df.to_parquet(MACRO_FILE)
    return macro_df