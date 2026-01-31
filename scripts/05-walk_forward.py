import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from pathlib import Path
import sys
import os
import shutil
from warnings import simplefilter

# Ignorar advertencias de fechas pandas
simplefilter(action="ignore", category=UserWarning)
simplefilter(action="ignore", category=FutureWarning)

# Configuración de Rutas
sys.path.append(str(Path(__file__).parent.parent))
from src.core.config import (
    CLUSTERS_FILE, MACRO_FILE, BENCHMARK_DIR, RESULTS_DIR, N_CLUSTERS
)
from src.envs.portfolio_env import PortfolioEnv
from src.models.agent_rl import create_agent

# Estilos Gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.1)

# ==========================================
# 1. MOTOR DE REPORTING (ROBUSTO)
# ==========================================
class ReportingEngine:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, df, benchmark_series):
        # --- FIX CRÍTICO: Asegurar que 'date' es datetime ---
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # ----------------------------------------------------

        equity = df['equity']
        returns = df['return']
        
        # Alinear Benchmark
        b_series = benchmark_series.reindex(returns.index).fillna(0)
        
        # Métricas
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(252)
        
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252)
        
        rf = 0.03
        sharpe = (cagr - rf) / vol if vol > 0 else 0
        sortino = (cagr - rf) / downside_vol if downside_vol > 0 else 0
        
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()
        
        # Alpha/Beta
        import scipy.stats as stats
        if len(returns) > 1 and len(b_series) > 1 and b_series.std() > 0:
            slope, intercept, _, _, _ = stats.linregress(b_series, returns)
            beta, alpha = slope, intercept * 252
        else:
            beta, alpha = 0, 0
            
        return {
            "Total Return": total_ret, "CAGR": cagr, "Vol": vol,
            "Max DD": max_dd, "Sharpe": sharpe, "Sortino": sortino,
            "Alpha": alpha, "Beta": beta, "Win Rate": len(returns[returns>0])/len(returns)
        }, drawdown, b_series

    def generate_report(self, df, bench, subfolder_name, title_prefix):
        """Genera reporte en una subcarpeta específica."""
        target_dir = self.base_dir / subfolder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Métricas (Aquí dentro se corrige la fecha automáticamente)
        metrics, dd, b_aligned = self.calculate_metrics(df, bench)
        
        # TXT
        with open(target_dir / "Executive_Summary.txt", "w") as f:
            f.write(f"--- REPORT: {title_prefix} ---\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    val = f"{v:.2%}" if k not in ["Sharpe", "Sortino", "Beta", "Vol"] else f"{v:.2f}"
                    if k == "Vol": val = f"{v:.2%}" # Excepción estética
                else:
                    val = str(v)
                f.write(f"{k:<15} | {val:>10}\n")
                
        # Gráficos (Usamos el DF ya procesado dentro de calculate_metrics? 
        # No, pasamos el original, así que lo corregimos al vuelo para plotear)
        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
        
        self._plot_performance(df, b_aligned, dd, target_dir, title_prefix)
        self._plot_allocation(df, target_dir, title_prefix)
        self._plot_dist(df, target_dir, title_prefix)
        
        return metrics

    def _plot_performance(self, df, bench, dd, folder, title):
        # Asegurar índice fecha para ploteo
        if 'date' in df.columns: df = df.set_index('date')
            
        eq = df['equity'] / df['equity'].iloc[0] * 100
        b_eq = (1 + bench).cumprod()
        b_eq = b_eq / b_eq.iloc[0] * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(eq, label='Athen AI', color='#1f77b4', linewidth=1.5)
        ax1.plot(b_eq, label='Benchmark', color='gray', linestyle='--', alpha=0.7)
        ax1.set_title(f"{title} - Performance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
        ax2.plot(dd, color='red', linewidth=0.8)
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(folder / "1_Performance.png")
        plt.close()

    def _plot_allocation(self, df, folder, title):
        if 'date' in df.columns: df = df.set_index('date')
        
        w_cols = [c for c in df.columns if c.endswith('_w') and 'cash' not in c]
        if not w_cols: return
        
        mean_w = df[w_cols].mean().sort_values(ascending=False).index
        plt.figure(figsize=(10, 5))
        plt.stackplot(df.index, df[mean_w].T, labels=mean_w, alpha=0.8, cmap='tab20')
        plt.plot(df.index, df['cash_weight'], label='CASH', color='k', linestyle=':', linewidth=1)
        plt.title(f"{title} - Allocation")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(folder / "2_Allocation.png")
        plt.close()

    def _plot_dist(self, df, folder, title):
        plt.figure(figsize=(8, 5))
        sns.histplot(df['return'], bins=50, kde=True, stat='density', color='blue', alpha=0.5)
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f"{title} - Returns Distribution")
        plt.savefig(folder / "3_Distribution.png")
        plt.close()

    def analyze_clusters(self, clusters_df):
        stats = []
        for c_id in range(N_CLUSTERS):
            col = f"C{c_id}_Ret"
            if col in clusters_df.columns:
                mu = clusters_df[col].mean() * 252
                sigma = clusters_df[col].std() * np.sqrt(252)
                stats.append({'Cluster': f"C{c_id}", 'Return': mu, 'Vol': sigma})
        
        if stats:
            sdf = pd.DataFrame(stats)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=sdf, x='Vol', y='Return', hue='Cluster', s=200, style='Cluster', palette='deep')
            for i in range(len(sdf)):
                plt.text(sdf.Vol[i], sdf.Return[i], sdf.Cluster[i], fontsize=9)
            plt.title("Cluster Profiles (Full History)")
            plt.grid(True, alpha=0.3)
            plt.savefig(self.base_dir / "0_Cluster_Profiles.png")
            plt.close()

# ==========================================
# 2. CARGA DE DATOS (FIXED)
# ==========================================
def load_clean_data():
    print("⏳ Cargando datos...")
    clusters = pd.read_parquet(CLUSTERS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    
    # Limpieza índices
    clusters.index = pd.to_datetime(clusters.index).normalize()
    macro.index = pd.to_datetime(macro.index).normalize()
    
    # Deduplicación
    clusters = clusters.groupby(clusters.index).last()
    macro = macro.groupby(macro.index).last()
    
    # Benchmark Mensual
    b_path = BENCHMARK_DIR / 'benchmark_monthly_returns.csv'
    if b_path.exists():
        print("   ✅ Usando Benchmark MENSUAL")
        b_df = pd.read_csv(b_path)
        # Fix warning parse
        b_df['date'] = pd.to_datetime(b_df['month'], format='%Y-%m') + pd.offsets.MonthEnd(0)
        b_df = b_df.set_index('date').sort_index()
        
        monthly_equity = (1 + b_df['monthly_return']).cumprod()
        daily_equity = monthly_equity.reindex(clusters.index, method='ffill').fillna(1.0)
        b = daily_equity.pct_change().fillna(0)
    else:
        print("   ⚠️ Usando Benchmark CERO")
        b = pd.Series(0, index=clusters.index)
        
    common = clusters.index.intersection(macro.index).intersection(b.index)
    return clusters.loc[common], macro.loc[common], b.loc[common]

def filter_dates(df, start, end):
    mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
    return df[mask]

# ==========================================
# 3. MOTOR WALK-FORWARD
# ==========================================
def run_walk_forward():
    # Setup carpetas
    wf_dir = RESULTS_DIR / "walk_forward_results"
    if wf_dir.exists(): shutil.rmtree(wf_dir)
    wf_dir.mkdir(parents=True)
    
    reporter = ReportingEngine(wf_dir)
    
    # Carga Datos
    c_full, m_full, b_full = load_clean_data()
    reporter.analyze_clusters(c_full) # Gráfico global de clusters
    
    # Ventanas
    windows = [
        ("2020-06-01", "2021-12-31", "2022-01-01", "2022-12-31"), # Fold 2022
        ("2020-06-01", "2022-12-31", "2023-01-01", "2023-12-31"), # Fold 2023
        ("2020-06-01", "2023-12-31", "2024-01-01", "2024-12-31"), # Fold 2024
    ]
    
    all_oos = []
    summary_data = []

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        fold_name = f"Fold_{te_start[:4]}"
        fold_path = wf_dir / fold_name
        fold_path.mkdir()
        
        print(f"\n📍 {fold_name}: Train({tr_start}->{tr_end}) Test({te_start}->{te_end})")
        
        # Datos Slicing
        c_tr = filter_dates(c_full, tr_start, tr_end)
        m_tr = filter_dates(m_full, tr_start, tr_end)
        b_tr = filter_dates(b_full, tr_start, tr_end)
        
        c_te = filter_dates(c_full, te_start, te_end)
        m_te = filter_dates(m_full, te_start, te_end)
        b_te = filter_dates(b_full, te_start, te_end)
        
        if len(c_te) == 0: continue
            
        # 1. ENTRENAMIENTO
        print(f"   🏋️ Entrenando ({len(c_tr)} días)...")
        env_train = PortfolioEnv(c_tr, m_tr, b_tr, leverage=2.0)
        model = create_agent(env_train, verbose=0)
        model.learn(total_timesteps=80000)
        model.save(fold_path / "model")
        
        # 2. VALIDACIÓN EN TRAIN (Generar Reporte Train)
        print("   🔍 Generando Reporte Train...")
        obs, _ = env_train.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_train.step(action)
        
        path_tr_log = env_train.save_execution_to_csv(filename_prefix="exec_train")
        df_tr_log = pd.read_csv(path_tr_log)
        
        # Reporte Train
        reporter_fold = ReportingEngine(fold_path)
        m_tr_res = reporter_fold.generate_report(df_tr_log, b_tr, "Train", f"{fold_name} - In-Sample")
        
        # 3. TEST (OOS)
        print("   🧪 Generando Reporte Test...")
        env_test = PortfolioEnv(c_te, m_te, b_te, leverage=2.0, log_dir=fold_path)
        obs, _ = env_test.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_test.step(action)
            
        path_te_log = env_test.save_execution_to_csv(filename_prefix="exec_test")
        df_te_log = pd.read_csv(path_te_log)
        all_oos.append(df_te_log)
        
        # Reporte Test
        m_te_res = reporter_fold.generate_report(df_te_log, b_te, "Test", f"{fold_name} - Out-of-Sample")
        
        summary_data.append({
            "Fold": fold_name,
            "Train_Ret": m_tr_res['Total Return'], "Test_Ret": m_te_res['Total Return'],
            "Train_Sharpe": m_tr_res['Sharpe'],    "Test_Sharpe": m_te_res['Sharpe']
        })

    # --- AGREGACIÓN FINAL ---
    if all_oos:
        print("\n🔗 Generando Curva Maestra (Stitched)...")
        full_oos = pd.concat(all_oos).sort_values('date').set_index('date')
        full_oos.index = pd.to_datetime(full_oos.index)
        full_oos['wf_equity'] = (1 + full_oos['return']).cumprod() * 100
        
        b_oos = filter_dates(b_full, full_oos.index.min(), full_oos.index.max())
        b_equity = (1 + b_oos).cumprod() * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(full_oos['wf_equity'], label='Athen AI (WF)', color='blue')
        plt.plot(b_equity, label='Benchmark', color='gray', linestyle='--')
        plt.title("Walk-Forward Global Performance")
        plt.legend()
        plt.savefig(wf_dir / "Global_WF_Chart.png")
        
        pd.DataFrame(summary_data).set_index("Fold").to_csv(wf_dir / "Metrics_Summary.csv")
        print(f"\n✅ Proceso completado. Revisa: {wf_dir}")

if __name__ == "__main__":
    run_walk_forward()