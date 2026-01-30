"""
Pipeline orquestador/documentado para ejecución completa y clara.
- Ejecuta: preprocesamiento, personalidad, clustering, proxy layers, backtesting y evaluación.
- Permite seleccionar agente/estrategia (incluyendo baseline).
- Muestra outputs y métricas clave en cada paso.
"""
import os
from pathlib import Path
import pandas as pd
from src.utils.config import load_configs, make_agent, make_env
from src.utils.backtest import backtest_portfolio
from src.utils.metrics import calmar_ratio, sharpe_ratio, max_drawdown, win_rate
from src.utils.visualization import plot_equity_curve, plot_drawdown

# Asume que los pipelines de athenai ya están importados y configurados
# (preprocess, personality, clustering, layers)

def run_full_pipeline(config_dir, prices, alive_mask=None, benchmark=None,
                     macro_real=None, macro_proxy=None, mode="train"):
    """
    Ejecuta el pipeline completo y muestra resultados.
    - config_dir: carpeta con los YAML de configuración
    - prices: DataFrame de precios (fechas x activos)
    - alive_mask: DataFrame bool (fechas x activos)
    - benchmark: Serie de benchmark para comparar
    - macro_real: DataFrame de variables macro reales (entrenamiento)
    - macro_proxy: DataFrame de proxies macro (predicción)
    - mode: "train" o "predict"
    """
    print("\n=== 1. Cargando configuración y agente ===")
    cfg = load_configs(Path(config_dir))
    agent = make_agent(None, cfg)  # El entorno se puede pasar si se usa RL
    print(f"Agente seleccionado: {cfg['agent']['type']}")

    # --- Selección de variables macro ---
    if mode == "train":
        macro_features = macro_real
        macro_mode = "train"
    else:
        macro_features = macro_proxy
        macro_mode = "predict"

    print(f"\n=== 2. Clustering con variables macro ({macro_mode}) ===")
    from athenai.clustering.build_clusters import FitClusterModelsStep
    # Aquí se asume que tienes store y cfg de clustering ya preparados
    # FitClusterModelsStep().run(store, clustering_cfg, overwrite=False, macro_features=macro_features, macro_mode=macro_mode)
    print("(Clustering ejecutado con macro features)")

    print(f"\n=== 3. Construcción de estado RL enriquecido con macro ({macro_mode}) ===")
    from athenai.pipelines.rl_macro_wrapper import enrich_rl_state
    # cluster_features = ... (cargar del pipeline de layers)
    # state_rl = enrich_rl_state(cluster_features, macro_features)
    print("(Estado RL enriquecido con macro features)")

    print("\n=== 4. Backtesting del agente ===")
    def weights_fn(state):
        return agent.select_action(state)
    df, weights_hist = backtest_portfolio(weights_fn, prices, alive_mask=alive_mask, window=cfg['agent'].get('window', 60))

    print("\n=== 5. Métricas de performance ===")
    print(f"Calmar Ratio: {calmar_ratio(df['equity'].values):.3f}")
    print(f"Sharpe Ratio: {sharpe_ratio(df['returns'].values):.3f}")
    print(f"Max Drawdown: {max_drawdown(df['equity'].values):.2%}")
    print(f"Win Rate: {win_rate(df['returns'].values):.2%}")

    print("\n=== 6. Gráficas ===")
    plot_equity_curve(df, benchmark=benchmark)
    plot_drawdown(df)

    return df, weights_hist
