import numpy as np
import pandas as pd

def backtest_portfolio(weights_fn, prices, alive_mask=None, initial_cash=1.0, freq='D', **kwargs):
    """
    Simula la evolución de un portafolio dado un generador de pesos (weights_fn) y precios.
    - weights_fn: función (state) -> pesos (n_assets,)
    - prices: DataFrame (fechas x activos)
    - alive_mask: DataFrame (fechas x activos, bool), opcional
    - initial_cash: capital inicial
    - freq: frecuencia de rebalanceo
    - kwargs: argumentos extra para weights_fn
    Devuelve: DataFrame con equity curve, retornos, drawdown, etc.
    """
    dates = prices.index
    n_assets = prices.shape[1]
    equity = [initial_cash]
    weights_hist = []
    returns_hist = []
    state = {}
    for t in range(1, len(dates)):
        # Estado: retornos pasados, activos vivos
        window = kwargs.get('window', 60)
        start = max(0, t-window)
        state['returns'] = np.log(prices.iloc[start:t].values[1:] / prices.iloc[start:t].values[:-1])
        if alive_mask is not None:
            state['alive_mask'] = alive_mask.iloc[t-1].values
        else:
            state['alive_mask'] = ~np.isnan(prices.iloc[t-1].values)
        # Pesos
        weights = weights_fn(state)
        weights_hist.append(weights)
        # Retorno del portafolio
        ret = np.nansum(weights * (prices.iloc[t].values / prices.iloc[t-1].values - 1))
        returns_hist.append(ret)
        equity.append(equity[-1] * (1 + ret))
    df = pd.DataFrame({
        'date': dates[1:],
        'equity': equity[1:],
        'returns': returns_hist
    })
    df.set_index('date', inplace=True)
    df['drawdown'] = (df['equity'] / df['equity'].cummax()) - 1
    return df, np.array(weights_hist)

def calmar_ratio(equity_curve):
    max_dd = np.abs(np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1))
    cagr = (equity_curve[-1] / equity_curve[0]) ** (252/len(equity_curve)) - 1
    return cagr / max_dd if max_dd > 0 else np.nan

def sharpe_ratio(returns, freq=252):
    mean = np.nanmean(returns)
    std = np.nanstd(returns)
    return (mean * np.sqrt(freq)) / std if std > 0 else np.nan

def max_drawdown(equity_curve):
    return np.abs(np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1))

def win_rate(returns):
    return np.mean(np.array(returns) > 0)
