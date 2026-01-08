import pandas as pd
import numpy as np

def calculate_metrics(df):
    """
    Calculates Momentum (3-month return) and Volatility (std dev of daily returns).
    Assumes df has 'close' column and is indexed by datetime.
    """
    # Resample to daily (if not already) or just use the provided frequency
    # 007XY.csv seems to have 4-hour candles approximately.
    # Let's resample to Daily for volatility calculation standard.
    
    df_daily = df.resample('D').last().dropna()
    
    # Calculate Returns
    df_daily['returns'] = df_daily['close'].pct_change()
    
    # 3-Month Momentum: Return over last 90 days approx
    # Simple approach: (Last Price / Price 90 days ago) - 1
    # Check if we have enough data
    if len(df_daily) < 90:
        return None, None
        
    price_now = df_daily['close'].iloc[-1]
    price_90d_ago = df_daily['close'].iloc[-90]
    momentum = (price_now / price_90d_ago) - 1
    
    # Volatility: Std dev of daily returns over last 90 days
    last_90_returns = df_daily['returns'].tail(90)
    volatility = last_90_returns.std() * np.sqrt(252) # Annualized
    
    return momentum, volatility

def rank_algorithms(algo_data_dict):
    """
    Rank algorithms based on Momentum (High is good) and Volatility (Low is good).
    Simple Score = Momentum / Volatility (Sharpe-like) or just Rank Sum.
    Instructions say: "Top 20 by Momentum (3m) and Low Volatility"
    Let's filter for valid volatility and then sort.
    """
    results = []
    
    for algo_name, data in algo_data_dict.items():
        mom, vol = calculate_metrics(data)
        if mom is not None and vol is not None and vol > 0:
            results.append({
                'algo': algo_name,
                'momentum': mom,
                'volatility': vol,
                'ratio': mom / vol # Simple risk-adjusted return
            })
    
    if not results:
        return []
        
    df_results = pd.DataFrame(results)
    
    # Strategy: High Momentum, Low Volatility.
    # We can sort by the Ratio (Sharpe-ish)
    df_results = df_results.sort_values(by='ratio', ascending=False)
    
    return df_results.head(20)['algo'].tolist()
