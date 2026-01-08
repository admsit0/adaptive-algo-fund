import time
import pandas as pd
from data_loader import DataLoader
from strategy_baseline import rank_algorithms
from benchmark_analysis import analyze_benchmark
import os

def main():
    start_time = time.time()
    
    print("=== Week 1 Baseline Strategy Execution ===")
    data_dir = os.path.join(os.getcwd(), 'data')
    
    # 1. Analyze Benchmark
    analyze_benchmark(data_dir)
    
    # 2. Initialize Data Loader
    loader = DataLoader(data_dir)
    
    # 3. Get Availability Mask (Active Algos)
    # Using 2024 as the availability year based on our file inspection (007XY.csv had data up to Feb 2024)
    # Ideally logic should not hardcode "2024" but detect "recent" relative to dataset.
    # But for this baseline, we verify availability.
    active_algos = loader.get_availability_mask()
    print(f"Active Algorithms found: {len(active_algos)}")
    
    # 4. Load Data & Calculate Metrics
    # We need to process active algos. Loading all 14k might be slow.
    # Let's target a subset if there are too many, or optimize loader.
    # The requirement is to be under 10 minutes.
    
    algo_data_map = {}
    
    # Limit processing for safety if getting active_algos takes too long or returns too many.
    # However, to be fair, we should try to process as many as possible.
    # If len(active_algos) is huge, we might hit memory limits reading all into a dict.
    # Better approach: Calculate metrics iteratively to save memory.
    
    results = []
    
    print("Calculating metrics for active algorithms...")
    count = 0
    from strategy_baseline import calculate_metrics
    
    for filename in active_algos:
        try:
            # OPTIMIZATION: Use load_tail_df to read only last 150 lines
            # This is sufficient for 90-day momentum/volatility (approx 90 lines if daily, 
            # if intraday, we might need more. But load_tail_df default 150 lines is good start.
            # wait, 007XY had 4h candles. 6 candles * 90 days = 540 lines!
            # I should increase n_lines.
            
            # 6 candles per day * 90 days = 540. Let's start with 600.
            # Strategy: daily resample.
            df = loader.load_tail_df(filename, n_lines=600)
            
            # calculate_metrics handles empty/short dfs
            mom, vol = calculate_metrics(df)
            
            if mom is not None and vol is not None and vol > 0:
                results.append({
                    'algo': filename.replace('.csv', ''),
                    'momentum': mom,
                    'volatility': vol,
                    'ratio': mom / vol
                })
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            pass
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(active_algos)}...")
            
    print(f"Total processed: {count}")
    
    # 5. Rank and Select
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='ratio', ascending=False)
        top_20 = df_results.head(20)
        
        print("\n=== Top 20 Algorithms ===")
        print(top_20[['algo', 'momentum', 'volatility', 'ratio']])
        
        # Save results
        top_20.to_csv('week1_baseline_selection.csv', index=False)
        print("\nSelection saved to week1_baseline_selection.csv")
    else:
        print("No valid algorithms found.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal Execution Time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
