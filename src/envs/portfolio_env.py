import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path

class PortfolioEnv(gym.Env):
    def __init__(self, cluster_df, macro_df, benchmark_ret, window=30, log_dir="logs", leverage=2.0, turnover_penalty=0.005):
        super().__init__()
        
        # Validación de integridad
        if len(cluster_df) != len(benchmark_ret):
            raise ValueError(f"❌ Desalineación: Clusters({len(cluster_df)}) vs Bench({len(benchmark_ret)})")
        
        self.dates = cluster_df.index
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convertir a float32 para la red neuronal
        self.data = cluster_df.values.astype(np.float32)
        self.macro = macro_df.values.astype(np.float32)
        self.bench = benchmark_ret.values.astype(np.float32)
        
        self.leverage = leverage
        self.turnover_penalty = turnover_penalty
        
        # Auto-detectar features (Ret, Vol, Mom)
        self.n_features_per_asset = 3 
        self.n_assets = self.data.shape[1] // self.n_features_per_asset
        
        self.window = window
        self.current_step = window
        
        # Espacios de Acción y Observación
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        obs_dim = self.data.shape[1] + self.macro.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window, obs_dim), dtype=np.float32)
        
        # Estado
        self.portfolio_value = 100.0
        self.prev_weights = np.zeros(self.n_assets + 1)
        self.prev_weights[-1] = 1.0 # 100% Cash inicial
        self.execution_log = []
        self.episode_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.portfolio_value = 100.0
        self.prev_weights = np.zeros(self.n_assets + 1)
        self.prev_weights[-1] = 1.0
        self.execution_log = []
        self.episode_id += 1
        return self._get_obs(), {}

    def _get_obs(self):
        d = self.data[self.current_step - self.window : self.current_step]
        m = self.macro[self.current_step - self.window : self.current_step]
        return np.concatenate([d, m], axis=1)

    def step(self, action):
        # 1. Normalizar Pesos (Allocation relativa)
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        # 2. Aplicar Apalancamiento (Aggressive Mode)
        # Inversión real = Pesos relativos * leverage
        w_invest_relative = weights[:-1]
        w_invest_leveraged = w_invest_relative * self.leverage
        
        # 3. Datos del día
        indices_ret = [i * self.n_features_per_asset for i in range(self.n_assets)]
        r_clusters = self.data[self.current_step, indices_ret]
        r_bench = self.bench[self.current_step]
        
        # 4. Cálculo P&L y Costes
        # Turnover real: cuánto capital muevo realmente (incluyendo deuda)
        real_turnover = np.sum(np.abs((weights - self.prev_weights) * self.leverage))
        cost = real_turnover * 0.0005 # 5 bps
        
        gross_ret = np.dot(w_invest_leveraged, r_clusters)
        port_ret = gross_ret - cost
        
        # 5. Reward (Sortino Agresivo)
        alpha = port_ret - r_bench
        reward = alpha * 10 
        
        # Penalización Drawdown (Solo si es severo > 2%)
        if port_ret < -0.02: 
            reward -= abs(port_ret) * 50
            
        # Penalización Turnover (Muy baja para permitir agresividad)
        reward -= real_turnover * self.turnover_penalty
        
        reward = np.clip(reward, -10, 10)

        # 6. Update
        self.portfolio_value *= (1 + port_ret)
        self.prev_weights = weights
        
        # 7. Logging Detallado
        log_entry = {
            "date": self.dates[self.current_step],
            "equity": self.portfolio_value,
            "return": port_ret,
            "benchmark_return": r_bench,
            "leverage": self.leverage,
            "cost": cost,
            "cash_weight": weights[-1]
        }
        # Guardar pesos por cluster
        for i, w in enumerate(w_invest_relative):
            log_entry[f"C{i}_w"] = w
            
        self.execution_log.append(log_entry)
        
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        
        return self._get_obs(), reward, terminated, False, {'ret': port_ret}

    def save_execution_to_csv(self, filename_prefix="train"):
        if not self.execution_log: return None
        df = pd.DataFrame(self.execution_log)
        path = self.log_dir / f"{filename_prefix}_episode_{self.episode_id}.csv"
        df.to_csv(path, index=False)
        return path