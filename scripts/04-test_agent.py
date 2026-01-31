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
    CLUSTERS_FILE, MACRO_FILE, BENCHMARK_DIR, RESULTS_DIR,
    TEST_START_DATE, TEST_END_DATE, N_CLUSTERS
)
from src.envs.portfolio_env import PortfolioEnv


# ==========================================
# 1. MOTOR DE REPORTING
# ==========================================
class ReportingEngine:
    """Clase encargada de generar métricas y gráficos con rigor institucional."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, df, benchmark_series):
        """Calcula KPIs financieros (Sortino, Sharpe, Calmar, etc.)."""
        # Preparar datos
        df = df.set_index('date').sort_index()
        equity = df['equity']
        returns = df['return']
        
        # Benchmark alineado
        b_series = benchmark_series.reindex(returns.index).fillna(0)
        
        # Métricas Core
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(252)
        
        # Downside Risk (para Sortino)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252)
        
        # Ratios
        rf = 0.03 # Risk Free 3%
        sharpe = (cagr - rf) / vol if vol > 0 else 0
        sortino = (cagr - rf) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Alpha / Beta
        import scipy.stats as stats
        if len(returns) > 1 and len(b_series) > 1 and b_series.std() > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(b_series, returns)
            beta = slope
            alpha = intercept * 252 # Anualizado
        else:
            beta, alpha = 0, 0
        
        return {
            "Period": f"{len(df)} days",
            "Total Return": total_ret,
            "CAGR": cagr,
            "Volatility": vol,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Alpha (Ann.)": alpha,
            "Beta": beta,
            "Win Rate": len(returns[returns > 0]) / len(returns)
        }, drawdown, b_series

    def generate_full_report(self, df_test, bench_series, prefix="Test"):
        """Genera el reporte completo (TXT + 4 Gráficos)."""
        print(f"📊 Generando Reporte Profesional para: {prefix}")
        
        # 1. Calcular Métricas
        metrics, dd, b_aligned = self.calculate_metrics(df_test, bench_series)
        
        # 2. Guardar Resumen TXT
        report_path = self.output_dir / f"{prefix}_Executive_Report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- ATHEN AI: EXECUTIVE REPORT ({prefix}) ---\n\n")
            f.write(f"Period: {df_test['date'].iloc[0]} -> {df_test['date'].iloc[-1]}\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    val = f"{v:.2%}" if "Ratio" not in k and "Beta" not in k else f"{v:.2f}"
                else:
                    val = str(v)
                f.write(f"{k:<25} | {val:>10}\n")
            f.write("-" * 40 + "\n")
            
        print(f"   📄 Resumen guardado en: {report_path}")

        # 3. GENERAR GRÁFICOS
        self._plot_performance(df_test, b_aligned, dd, prefix)
        self._plot_allocation(df_test, prefix)
        self._plot_returns_dist(df_test, prefix)
        
        return metrics

    def _plot_performance(self, df, bench, dd, prefix):
        """Gráfico 1: Equity + Drawdown"""
        df = df.set_index('date')
        equity = df['equity'] / df['equity'].iloc[0] * 100
        b_eq = (1 + bench).cumprod()
        b_eq = b_eq / b_eq.iloc[0] * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(equity, label='Athen AI', color='#1f77b4', linewidth=2)
        ax1.plot(b_eq, label='Benchmark', color='gray', linestyle='--', alpha=0.7)
        ax1.set_title(f'{prefix}: Cumulative Performance (Base 100)', fontweight='bold')
        ax1.set_ylabel('Equity Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
        ax2.plot(dd, color='red', linewidth=1)
        ax2.set_title('Drawdown Profile', fontweight='bold')
        ax2.set_ylabel('% Drawdown')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_1_Performance.png")
        plt.close()

    def _plot_allocation(self, df, prefix):
        """Gráfico 2: Allocation Stacked"""
        df = df.set_index('date')
        w_cols = [c for c in df.columns if c.endswith('_w') and 'cash' not in c]
        
        if not w_cols:
            print(f"⚠️ Alerta: No se encontraron columnas de pesos para el gráfico.")
            return

        mean_w = df[w_cols].mean().sort_values(ascending=False).index
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(df.index, df[mean_w].T, labels=mean_w, alpha=0.85, cmap='tab20')
        plt.plot(df.index, df['cash_weight'], label='CASH', color='black', linestyle=':', linewidth=1.5)
        
        plt.title(f'{prefix}: Dynamic Cluster Allocation', fontweight='bold')
        plt.ylabel('Weight (0-1)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_2_Allocation.png")
        plt.close()

    def _plot_returns_dist(self, df, prefix):
        """Gráfico 3: Distribución"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['return'], bins=50, kde=True, color='blue', stat='density', alpha=0.5)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'{prefix}: Daily Return Distribution')
        plt.savefig(self.output_dir / f"{prefix}_3_Distribution.png")
        plt.close()

    def analyze_clusters_profile(self, clusters_df):
        """Gráfico 4: Perfil de Riesgo/Retorno de los Clusters"""
        print("🔍 Generando perfil de Clusters...")
        stats = []
        for c_id in range(N_CLUSTERS):
            col = f"C{c_id}_Ret"
            if col in clusters_df.columns:
                mu = clusters_df[col].mean() * 252
                sigma = clusters_df[col].std() * np.sqrt(252)
                stats.append({'Cluster': f"C{c_id}", 'Return': mu, 'Vol': sigma})
        
        if not stats: return

        sdf = pd.DataFrame(stats)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=sdf, x='Vol', y='Return', hue='Cluster', s=200, style='Cluster', palette='deep')
        
        for i in range(sdf.shape[0]):
            plt.text(sdf.Vol[i]+0.002, sdf.Return[i], sdf.Cluster[i], fontsize=10, fontweight='bold')
            
        plt.title('Cluster Risk/Return Profile (Annualized)', fontweight='bold')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "4_Cluster_Profiles.png")
        plt.close()

# ==========================================
# 2. GESTIÓN DE DATOS (FIXED & ROBUST)
# ==========================================
def load_and_audit_data(start_date, end_date):
    """Carga datos, DEDUPLICA y construye el Benchmark correctamente."""
    print("📂 Cargando y Auditando datos de Test...")
    
    # 1. Cargar Features
    clusters = pd.read_parquet(CLUSTERS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    
    # --- DEDUPLICACIÓN (CRÍTICA) ---
    # Convertimos a índice normalizado y nos quedamos con el último valor del día
    clusters.index = pd.to_datetime(clusters.index).normalize()
    macro.index = pd.to_datetime(macro.index).normalize()
    
    clusters = clusters.groupby(clusters.index).last()
    macro = macro.groupby(macro.index).last()
    
    # 2. Cargar Benchmark (MODO STRICT MENSUAL)
    # Evitamos trades_benchmark.csv para prevenir el 6000%
    b_path = BENCHMARK_DIR / 'benchmark_monthly_returns.csv'
    
    if b_path.exists():
        print("   ✅ Benchmark: Usando archivo MENSUAL (Fuente Limpia)")
        b_df = pd.read_csv(b_path)
        # Fecha = Fin de mes
        b_df['date'] = pd.to_datetime(b_df['month']) + pd.offsets.MonthEnd(0)
        b_df = b_df.set_index('date').sort_index()
        
        # A) Calcular Equity Curve Mensual (Base 1)
        monthly_equity = (1 + b_df['monthly_return']).cumprod()
        
        # B) Proyectar a Diario usando EL ÍNDICE DE CLUSTERS (Alineación Perfecta)
        # Esto repite el valor del mes anterior hasta que cambie el mes.
        daily_equity = monthly_equity.reindex(clusters.index, method='ffill')
        
        # Rellenar NaNs iniciales con 1.0 (si el benchmark empieza después, asumimos plano)
        daily_equity = daily_equity.fillna(method='bfill').fillna(1.0)
        
        # C) Sacar el retorno diario de esa curva alineada
        b_clean = daily_equity.pct_change().fillna(0)
        
    else:
        print("   ⚠️ No se encontró benchmark mensual. Usando 0.")
        b_clean = pd.Series(0, index=clusters.index)

    # 3. INTERSECCIÓN FINAL
    common = clusters.index.intersection(macro.index).intersection(b_clean.index)
    
    c_final = clusters.loc[common]
    m_final = macro.loc[common]
    b_final = b_clean.loc[common]

    # 4. FILTRO DE FECHAS (Test Range)
    mask = (c_final.index >= pd.to_datetime(start_date)) & (c_final.index <= pd.to_datetime(end_date))
    
    if not mask.any():
        raise ValueError(f"❌ El rango {start_date} -> {end_date} está vacío o sin datos.")

    c_test = c_final[mask]
    m_test = m_final[mask]
    b_test = b_final[mask]
    
    print(f"✅ Datos Test Listos: {len(c_test)} días ({start_date} - {end_date}).")
    return c_test, m_test, b_test, c_final

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
def run_test():
    # 1. Cargar Datos
    test_c, test_m, test_b, full_c = load_and_audit_data(TEST_START_DATE, TEST_END_DATE)
    
    # 2. Configurar Reporting
    log_dir = RESULTS_DIR / "test_results"
    reporter = ReportingEngine(log_dir)
    
    # Gráfico 4: Perfil de Clusters
    reporter.analyze_clusters_profile(full_c)
    
    # 3. Cargar Agente RL
    print("🤖 Cargando Athen AI...")
    model_path = "ppo_athenai_final"

    model = PPO.load(model_path)
    
    # 4. Ejecutar Entorno de Test
    print(f"   🧪 Ejecutando Backtest ({len(test_c)} steps)...")
    env = PortfolioEnv(test_c, test_m, test_b, log_dir=log_dir, leverage=2.0)
    
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    csv_path = env.save_execution_to_csv(filename_prefix="final_test_execution")
    
    # 5. Generar Reporte Final
    if csv_path:
        rl_df = pd.read_csv(csv_path)
        rl_df['date'] = pd.to_datetime(rl_df['date'])
        
        metrics = reporter.generate_full_report(rl_df, test_b, prefix="Final_Test")
        
        print("\n🏆 RESULTADOS FINALES:")
        print(f"   Retorno Total: {metrics['Total Return']:.2%}")
        print(f"   Sharpe Ratio:  {metrics['Sharpe Ratio']:.2f}")
        print(f"   Max Drawdown:  {metrics['Max Drawdown']:.2%}")
        print(f"   Reportes guardados en: {log_dir}")

if __name__ == "__main__":
    run_test()
    