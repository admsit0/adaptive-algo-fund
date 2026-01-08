import pandas as pd
import numpy as np
import os

def analyze_benchmark(data_dir):
    trades_path = os.path.join(data_dir, 'trades_benchmark.csv')
    monthly_returns_path = os.path.join(data_dir, 'benchmark_monthly_returns.csv')
    
    print(f"Loading benchmark data from {data_dir}...")
    
    # Analyze Trades for AUM and Strategy Type
    if os.path.exists(trades_path):
        trades = pd.read_csv(trades_path)
        print("Trades data loaded.")
        
        # Calculate Latent AUM
        # Assuming AUM is listed directly or can be inferred. The instruction says "Latent AUM".
        # Let's look at the 'AUM' column if it exists.
        if 'AUM' in trades.columns:
            max_aum = trades['AUM'].max()
            mean_aum = trades['AUM'].mean()
            print(f"Max AUM: {max_aum:,.2f}")
            print(f"Mean AUM: {mean_aum:,.2f}")
        
    # Analyze Monthly Returns for Drawdown
    if os.path.exists(monthly_returns_path):
        returns = pd.read_csv(monthly_returns_path)
        print("Monthly returns loaded.")
        
        # Calculate Drawdown
        # Assuming 'Equity End' or similar column tracks value over time
        if 'Equity End' in returns.columns:
            equity = returns['Equity End']
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            max_drawdown = drawdown.min()
            print(f"Max Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    analyze_benchmark('data')
