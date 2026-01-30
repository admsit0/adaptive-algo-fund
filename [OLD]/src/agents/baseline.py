import numpy as np
from typing import Any, Dict
import gym
from src.agents.base import BaseAgent

class BaselineAgent(BaseAgent):
    """
    Baseline agent: invierte en los N mejores activos vivos según rendimiento pasado.
    - N configurable (por defecto 10)
    - Pesos iguales o dinámicos (por defecto iguales)
    - Rebalanceo configurable (por defecto cada día)
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: Dict[str, Any]):
        super().__init__(observation_space, action_space, config)
        self.N = config.get('N', 10)
        self.weighting = config.get('weighting', 'equal')  # 'equal' o 'dynamic'
        self.window = config.get('window', 60)  # días para ranking
        self.last_selected = None
        self.last_weights = None

    def select_action(self, state: Any) -> Any:
        """
        Espera que el estado contenga:
        - 'returns': matriz (window, n_assets) de retornos recientes
        - 'alive_mask': vector booleano de activos vivos
        """
        returns = state['returns']  # shape: (window, n_assets)
        alive = state['alive_mask']  # shape: (n_assets,)
        n_assets = returns.shape[1]

        # Calcular rendimiento acumulado en ventana
        perf = np.nansum(returns, axis=0)
        perf[~alive] = -np.inf  # penalizar inactivos
        top_idx = np.argsort(perf)[-self.N:][::-1]

        weights = np.zeros(n_assets)
        if self.weighting == 'equal':
            weights[top_idx] = 1.0 / self.N
        elif self.weighting == 'dynamic':
            pos_perf = np.clip(perf[top_idx], a_min=0, a_max=None)
            if np.sum(pos_perf) > 0:
                weights[top_idx] = pos_perf / np.sum(pos_perf)
            else:
                weights[top_idx] = 1.0 / self.N
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

        self.last_selected = top_idx
        self.last_weights = weights
        return weights

    def observe(self, state, action, reward, next_state, done):
        pass
    def update(self):
        pass
    def save(self, path: str):
        pass
    def load(self, path: str):
        pass
