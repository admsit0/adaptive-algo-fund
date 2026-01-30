import numpy as np
import pandas as pd

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
