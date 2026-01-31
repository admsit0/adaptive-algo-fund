import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
filterwarnings("ignore")


from src.core.config import (
    CLUSTERS_FILE, MACRO_FILE, BENCHMARK_DIR, TRAINING_LOG_DIR,
    TRAIN_START_DATE, TRAIN_END_DATE
)
from src.envs.portfolio_env import PortfolioEnv
from src.models.agent_rl import create_agent

def load_and_filter_data(start_date=None, end_date=None):
    """Carga datos maestros y filtra por fechas específicas."""
    print("⏳ Cargando datos...")
    
    # 1. Carga Cruda
    clusters = pd.read_parquet(CLUSTERS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    
    # Carga Benchmark con fallback
    b_path = os.path.join(BENCHMARK_DIR, 'trades_benchmark.csv')
    if not os.path.exists(b_path): b_path = os.path.join(BENCHMARK_DIR, 'benchmark_monthly_returns.csv')
    bench_df = pd.read_csv(b_path)
    
    # Procesar Benchmark
    if 'trades' in str(b_path):
        bench_df['dateClose'] = pd.to_datetime(bench_df['dateClose'], format='mixed', utc=True).dt.tz_localize(None).dt.normalize()
        bench_returns = bench_df.sort_values('dateClose').set_index('dateClose')['equity_EOD'].resample('D').last().ffill().pct_change().fillna(0)
    else:
        bench_returns = pd.Series(0, index=clusters.index)

    # 2. Limpieza de Índices (FIX DUPLICADOS)
    if clusters.index.tz is not None: clusters.index = clusters.index.tz_localize(None)
    clusters.index = clusters.index.normalize()
    macro.index = macro.index.normalize()
    
    # GroupBy last() elimina duplicados conservando el último dato válido
    clusters = clusters.groupby(clusters.index).last()
    macro = macro.groupby(macro.index).last()
    bench_returns = bench_returns.groupby(bench_returns.index).last()

    # 3. Intersección Global
    common = clusters.index.intersection(macro.index).intersection(bench_returns.index)
    c, m, b = clusters.loc[common], macro.loc[common], bench_returns.loc[common]
    
    # 4. Filtro de Fechas (Train/Val Split)
    if start_date:
        mask = (c.index >= pd.to_datetime(start_date))
        c, m, b = c[mask], m[mask], b[mask]
    if end_date:
        mask = (c.index <= pd.to_datetime(end_date))
        c, m, b = c[mask], m[mask], b[mask]

    # 5. Sanitización
    c = c.replace([np.inf, -np.inf], np.nan).fillna(0)
    m = m.replace([np.inf, -np.inf], np.nan).fillna(0)
    b = b.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return c, m, b

if __name__ == "__main__":
    # Usar fechas de config.py
    print(f"📅 Configurando Entrenamiento: {TRAIN_START_DATE} -> {TRAIN_END_DATE}")
    
    train_c, train_m, train_b = load_and_filter_data(TRAIN_START_DATE, TRAIN_END_DATE)
    
    print(f"✅ Datos Listos: {len(train_c)} días de entrenamiento.")
    print(f"📊 Check: Media Retornos: {train_c.mean().mean():.6f}")

    
    TO_PENALTY = 0.0005
    TIMESTEPS = 100000

    env = PortfolioEnv(
        train_c, train_m, train_b, 
        log_dir=TRAINING_LOG_DIR,
        leverage=3.0,            # Apalancamiento x3
        turnover_penalty=TO_PENALTY   # Coste bajo
    )
    
    model = create_agent(env)
    
    print("🚀 Iniciando Entrenamiento...")
    model.learn(total_timesteps=TIMESTEPS)
    
    model.save("ppo_athenai_final")
    print("💾 Modelo guardado: ppo_athenai_final.zip")

    env.save_execution_to_csv(f'ppo_{TIMESTEPS}_steps_{TO_PENALTY}_penalty')
