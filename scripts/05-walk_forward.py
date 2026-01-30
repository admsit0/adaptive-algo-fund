import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pathlib import Path
import sys
import os

# Configuración de Rutas
sys.path.append(str(Path(__file__).parent.parent))
from src.core.config import (
    CLUSTERS_FILE, MACRO_FILE, BENCHMARK_DIR, PROCESSED_DIR
)
from src.envs.portfolio_env import PortfolioEnv
from src.models.agent_rl import create_agent

# --- UTILIDADES DE CARGA LIMPIA (Reutilizadas para garantizar robustez) ---
def load_clean_data():
    """Carga los datos maestros y elimina duplicados."""
    print("⏳ Cargando datos maestros...")
    clusters = pd.read_parquet(CLUSTERS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    
    # Benchmark
    b_path = BENCHMARK_DIR / 'trades_benchmark.csv'
    if not b_path.exists(): b_path = BENCHMARK_DIR / 'benchmark_monthly_returns.csv'
    b_df = pd.read_csv(b_path)
    
    if 'trades' in str(b_path):
        b_df['dateClose'] = pd.to_datetime(b_df['dateClose'], format='mixed', utc=True).dt.tz_localize(None).dt.normalize()
        b = b_df.sort_values('dateClose').set_index('dateClose')['equity_EOD'].resample('D').last().ffill().pct_change().fillna(0)
    else:
        b = pd.Series(0, index=clusters.index)

    # Limpieza índices
    clusters.index = pd.to_datetime(clusters.index).normalize()
    macro.index = pd.to_datetime(macro.index).normalize()
    
    # Deduplicación crítica
    clusters = clusters.groupby(clusters.index).last()
    macro = macro.groupby(macro.index).last()
    b = b.groupby(b.index).last()
    
    # Intersección
    common = clusters.index.intersection(macro.index).intersection(b.index)
    return clusters.loc[common], macro.loc[common], b.loc[common]

def filter_dates(df, start, end):
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    return df[mask]

# --- MOTOR WALK-FORWARD ---
def run_walk_forward():
    print("🔄 INICIANDO WALK-FORWARD VALIDATION (Cross-Val Dinámico)")
    
    # 1. Configuración de Ventanas (Expanding Window)
    # Formato: (Inicio_Train, Fin_Train, Inicio_Test, Fin_Test)
    windows = [
        ("2020-06-01", "2021-12-31", "2022-01-01", "2022-12-31"), # Fold 1
        ("2020-06-01", "2022-12-31", "2023-01-01", "2023-12-31"), # Fold 2
        ("2020-06-01", "2023-12-31", "2024-01-01", "2024-12-31"), # Fold 3
    ]
    
    # Directorio de resultados
    wf_dir = PROCESSED_DIR / "walk_forward_results"
    wf_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos una vez
    c_full, m_full, b_full = load_clean_data()
    
    all_oos_results = [] # Aquí guardaremos los trozos de test
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        fold_name = f"Fold_{test_start[:4]}"
        print(f"\n📍 EJECUTANDO {fold_name}")
        print(f"   Train: {train_start} -> {train_end}")
        print(f"   Test:  {test_start} -> {test_end}")
        
        # A) Preparar Datos
        train_c = filter_dates(c_full, train_start, train_end)
        train_m = filter_dates(m_full, train_start, train_end)
        train_b = filter_dates(b_full, train_start, train_end)
        
        test_c = filter_dates(c_full, test_start, test_end)
        test_m = filter_dates(m_full, test_start, test_end)
        test_b = filter_dates(b_full, test_start, test_end)
        
        if len(test_c) == 0:
            print(f"⚠️ Saltando {fold_name}: No hay datos de test.")
            continue
            
        # B) Entrenar Modelo
        print(f"   🏋️ Entrenando Agente en {len(train_c)} días...")
        env_train = PortfolioEnv(train_c, train_m, train_b, leverage=2.0, turnover_penalty=0.005)
        model = create_agent(env_train, verbose=0)
        model.learn(total_timesteps=80000) # Menos steps porque re-entrenamos cada año
        
        # Guardar modelo del año
        model_path = wf_dir / f"ppo_{fold_name}"
        model.save(model_path)
        
        # C) Testear (OOS)
        print(f"   🧪 Testeando en {len(test_c)} días...")
        env_test = PortfolioEnv(test_c, test_m, test_b, log_dir=wf_dir, leverage=2.0)
        
        obs, _ = env_test.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_test.step(action)
            
        # Guardar log del test
        csv_path = env_test.save_execution_to_csv(filename_prefix=f"test_{fold_name}")
        
        if csv_path:
            df_res = pd.read_csv(csv_path)
            all_oos_results.append(df_res)
            
    # --- AGREGACIÓN DE RESULTADOS ---
    if not all_oos_results:
        print("❌ No se generaron resultados.")
        return

    print("\n🔗 Uniendo resultados Walk-Forward...")
    full_oos = pd.concat(all_oos_results)
    full_oos['date'] = pd.to_datetime(full_oos['date'])
    full_oos = full_oos.sort_values('date').set_index('date')
    
    # Recalcular Equity Curve Continua (Stitching)
    # Usamos los retornos diarios para reconstruir una curva desde 100
    full_oos['real_return'] = full_oos['return']
    full_oos['wf_equity'] = (1 + full_oos['real_return']).cumprod() * 100
    
    # Benchmark Acumulado en el mismo periodo
    # Filtramos el benchmark global para coincidir con las fechas del OOS total
    oos_start = full_oos.index.min()
    oos_end = full_oos.index.max()
    b_oos = filter_dates(b_full, oos_start, oos_end)
    b_equity = (1 + b_oos).cumprod() * 100
    
    # --- VISUALIZACIÓN FINAL ---
    plt.figure(figsize=(12, 6))
    plt.plot(full_oos.index, full_oos['wf_equity'], label='Athen AI (Walk-Forward)', color='blue', linewidth=2)
    plt.plot(b_equity.index, b_equity, label='Benchmark', color='gray', linestyle='--', alpha=0.7)
    
    # Marcar años
    for year in range(oos_start.year, oos_end.year + 1):
        plt.axvline(pd.Timestamp(f"{year}-01-01"), color='black', linestyle=':', alpha=0.3)
    
    plt.title(f"Walk-Forward Performance ({oos_start.date()} - {oos_end.date()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_chart = wf_dir / "walk_forward_chart.png"
    plt.savefig(output_chart)
    
    # Métricas Finales
    total_ret_wf = (full_oos['wf_equity'].iloc[-1] / 100) - 1
    total_ret_bm = (b_equity.iloc[-1] / 100) - 1
    
    print("\n🏆 RESULTADOS WALK-FORWARD:")
    print(f"   Athen AI Total Return: {total_ret_wf:.2%}")
    print(f"   Benchmark Total Return: {total_ret_bm:.2%}")
    print(f"   Gráfico guardado en: {output_chart}")
    
    # Guardar CSV consolidado
    full_oos.to_csv(wf_dir / "full_walk_forward_trades.csv")

if __name__ == "__main__":
    run_walk_forward()