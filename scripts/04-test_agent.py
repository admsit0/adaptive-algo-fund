import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from warnings import filterwarnings
from pathlib import Path
import sys
import os

filterwarnings("ignore")

# Configuración de estilos para gráficos profesionales
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)

# Configuración de rutas
sys.path.append(str(Path(__file__).parent.parent))
from src.core.config import (
    CLUSTERS_FILE, MACRO_FILE, BENCHMARK_DIR, PROCESSED_DIR,
    TEST_START_DATE, TEST_END_DATE
)
from src.envs.portfolio_env import PortfolioEnv
from src.models.baseline import LinearBaseline

# --- MÓDULO DE ANALÍTICA FINANCIERA ---

def calculate_advanced_metrics(equity_curve, benchmark_curve=None):
    """Calcula KPIs financieros avanzados."""
    returns = equity_curve.pct_change().dropna()
    
    # 1. Retorno y CAGR
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    # 2. Riesgo
    volatility = returns.std() * np.sqrt(252)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    
    # 3. Ratios
    risk_free = 0.03 # 3% libre de riesgo aprox
    sharpe = (cagr - risk_free) / volatility if volatility != 0 else 0
    sortino = (cagr - risk_free) / downside_vol if downside_vol != 0 else 0
    
    # 4. Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    # 5. Win/Loss Stats
    win_rate = len(returns[returns > 0]) / len(returns)
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf
    
    metrics = {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility (Ann.)": f"{volatility:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Win Rate": f"{win_rate:.2%}",
        "Profit Factor": f"{profit_factor:.2f}"
    }
    
    if benchmark_curve is not None:
        bench_ret = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) - 1
        metrics["Benchmark Return"] = f"{bench_ret:.2%}"
        metrics["Alpha"] = f"{(total_return - bench_ret):.2%}"

    return metrics, drawdown

def generate_professional_report(rl_df, bench_series, output_dir):
    """Genera un reporte PDF-style (PNGs + TXT) completo."""
    print("📊 Generando Reporte Avanzado...")
    
    # Preparar datos
    rl_df['date'] = pd.to_datetime(rl_df['date'])
    rl_df = rl_df.set_index('date')
    
    # Curvas Base 100
    curve_rl = rl_df['equity'] / rl_df['equity'].iloc[0] * 100
    
    # Alinear benchmark
    # Rellenamos huecos del benchmark si faltan días (ffill)
    bench_series = bench_series.reindex(rl_df.index).ffill().fillna(0)
    curve_bench = (1 + bench_series).cumprod()
    curve_bench = curve_bench / curve_bench.iloc[0] * 100
    
    # 1. Calcular Métricas
    metrics, dd_curve = calculate_advanced_metrics(curve_rl, curve_bench)
    
    # 2. Guardar Resumen TXT
    report_path = output_dir / "Executive_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*40 + "\n")
        f.write("   ATHEN AI - EXECUTIVE SUMMARY   \n")
        f.write("="*40 + "\n\n")
        f.write(f"Periodo de Análisis: {rl_df.index[0].date()} -> {rl_df.index[-1].date()}\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:<25} | {v:>10}\n")
        f.write("-" * 40 + "\n")
        
        # Análisis de Clusters
        w_cols = [c for c in rl_df.columns if "_weight" in c and "cash" not in c]
        if w_cols:
            avg_weights = rl_df[w_cols].mean().sort_values(ascending=False)
            f.write("\n\nTOP CLUSTERS (Exposición Media):\n")
            for c, w in avg_weights.head(5).items():
                f.write(f"  - {c}: {w:.1%}\n")
            f.write(f"  - Cash Promedio: {rl_df['cash_weight'].mean():.1%}\n")
            
    print(f"📄 Resumen ejecutivo guardado en: {report_path}")

    # 3. GRÁFICO 1: Performance & Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(curve_rl, label='Athen AI', color='#1f77b4', linewidth=2)
    ax1.plot(curve_bench, label='Benchmark', color='gray', linestyle='--', alpha=0.8)
    ax1.set_title('Cumulative Return (Base 100)', fontweight='bold')
    ax1.set_ylabel('Equity Value')
    ax1.legend()
    
    ax2.fill_between(dd_curve.index, dd_curve, 0, color='red', alpha=0.3)
    ax2.plot(dd_curve, color='red', linewidth=1)
    ax2.set_title('Drawdown Profile')
    ax2.set_ylabel('% Drawdown')
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_Performance_Overview.png")
    
    # 4. GRÁFICO 2: Allocation
    w_cols = [c for c in rl_df.columns if "_weight" in c and "cash" not in c]
    if w_cols:
        plt.figure(figsize=(12, 6))
        sorted_cols = rl_df[w_cols].mean().sort_values(ascending=False).index
        plt.stackplot(rl_df.index, rl_df[sorted_cols].T, labels=sorted_cols, alpha=0.8, cmap='tab20')
        plt.title('Dynamic Cluster Allocation', fontweight='bold')
        plt.ylabel('Weight (0-1)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
        plt.tight_layout()
        plt.savefig(output_dir / "2_Allocation_History.png")
        
    # 5. GRÁFICO 3: Distribución
    plt.figure(figsize=(10, 6))
    sns.histplot(rl_df['return'], bins=50, kde=True, color='blue', label='Athen AI', stat='density', alpha=0.4)
    plt.title('Distribución de Retornos Diarios')
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    plt.savefig(output_dir / "3_Return_Distribution.png")

# --- CARGA DE DATOS (FIX PRIORIDAD BENCHMARK) ---

def load_test_data(start_date, end_date):
    print("📂 Cargando datos de Test...")
    
    # 1. Cargar Datos Crudos
    clusters = pd.read_parquet(CLUSTERS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    
    # --- FIX: PRIORIDAD AL MENSUAL LIMPIO ---
    # Buscamos primero el mensual, si no, el trades (pero con cuidado)
    b_path_clean = BENCHMARK_DIR / 'benchmark_monthly_returns.csv'
    b_path_dirty = BENCHMARK_DIR / 'trades_benchmark.csv'
    
    if b_path_clean.exists():
        print("   ✅ Usando Benchmark Mensual (Limpio)")
        b_df = pd.read_csv(b_path_clean)
        # Parsear fecha 'YYYY-MM' -> Fin de Mes
        b_df['date'] = pd.to_datetime(b_df['month']) + pd.offsets.MonthEnd(0)
        b_df = b_df.set_index('date').sort_index()
        
        # Convertir retorno mensual a curva diaria
        # (1) Crear curva equity mensual
        monthly_equity = (1 + b_df['monthly_return']).cumprod()
        # (2) Resample a Diario y rellenar (ffill) para tener dato cada día
        daily_equity = monthly_equity.resample('D').ffill()
        # (3) Calcular retorno diario
        bench_returns = daily_equity.pct_change().fillna(0)
        
    elif b_path_dirty.exists():
        print("   ⚠️ AVISO: Usando Benchmark Trades (Puede contener depósitos)")
        b_df = pd.read_csv(b_path_dirty)
        b_df['dateClose'] = pd.to_datetime(b_df['dateClose'], format='mixed', utc=True).dt.tz_localize(None).dt.normalize()
        b_df = b_df.sort_values('dateClose')
        # Intentamos limpiar outliers masivos (>50%)
        bench_returns = b_df.set_index('dateClose')['equity_EOD'].resample('D').last().ffill().pct_change().fillna(0)
        bench_returns = bench_returns.mask(abs(bench_returns) > 0.5, 0)
    else:
        print("   ⚠️ No se encontró Benchmark. Usando ceros.")
        bench_returns = pd.Series(0, index=clusters.index)

    # 2. LIMPIEZA DE DUPLICADOS E ÍNDICES
    if clusters.index.tz is not None: clusters.index = clusters.index.tz_localize(None)
    clusters.index = clusters.index.normalize()
    macro.index = macro.index.normalize()
    if bench_returns.index.tz is not None: bench_returns.index = bench_returns.index.tz_localize(None)
    bench_returns.index = bench_returns.index.normalize()

    clusters = clusters.groupby(clusters.index).last()
    macro = macro.groupby(macro.index).last()
    bench_returns = bench_returns.groupby(bench_returns.index).last()

    # 3. INTERSECCIÓN (Reindexar Benchmark para no perder datos de Clusters)
    # En lugar de intersection() estricta (que puede fallar si al benchmark le falta 1 día),
    # forzamos al benchmark a adaptarse a los clusters.
    
    common_cm = clusters.index.intersection(macro.index)
    if len(common_cm) == 0: raise ValueError("❌ Error: Clusters y Macro no coinciden.")
    
    # Recortar Clusters y Macro
    c = clusters.loc[common_cm]
    m = macro.loc[common_cm]
    
    # Alinear Benchmark (Rellenar con 0 si faltan días)
    b = bench_returns.reindex(c.index).fillna(0)
    
    # 4. FILTRO DE FECHAS
    mask = (c.index >= pd.to_datetime(start_date)) & (c.index <= pd.to_datetime(end_date))
    
    if not mask.any():
        raise ValueError(f"❌ El rango {start_date} -> {end_date} está vacío.")

    c_test, m_test, b_test = c[mask], m[mask], b[mask]
    
    # Sanitizar
    c_test = c_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    m_test = m_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    b_test = b_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✅ Datos Test Listos: {len(c_test)} días.")
    return c_test, m_test, b_test

def run_test():
    # 1. Cargar
    test_c, test_m, test_b = load_test_data(TEST_START_DATE, TEST_END_DATE)
    
    # 2. Baseline
    print("📈 Calculando Baseline...")
    baseline = LinearBaseline()
    
    # 3. Agente RL
    print("🤖 Ejecutando Athen AI...")
    model_path = "ppo_athen_aggressive"
    if not os.path.exists(model_path + ".zip"):
        model_path = "ppo_athen_final"
        
    model = PPO.load(model_path)
    
    # Entorno Test
    log_dir = PROCESSED_DIR / "test_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    env = PortfolioEnv(test_c, test_m, test_b, log_dir=log_dir, leverage=2.0)
    
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    csv_path = env.save_execution_to_csv(filename_prefix="submission_trades")
    
    # 4. Reporte
    if csv_path:
        rl_df = pd.read_csv(csv_path)
        generate_professional_report(rl_df, test_b, log_dir)
        
        # Feedback rápido
        total_ret = (rl_df['equity'].iloc[-1] / rl_df['equity'].iloc[0]) - 1
        
        # Calcular retorno benchmark real en el periodo
        bench_cum = (1 + test_b).cumprod()
        bench_ret = bench_cum.iloc[-1] - 1
        
        print(f"\n🏆 RESULTADO TEST ({TEST_START_DATE} - {TEST_END_DATE}):")
        print(f"   Athen AI: {total_ret:.2%}")
        print(f"   Benchmark: {bench_ret:.2%}")

if __name__ == "__main__":
    run_test()