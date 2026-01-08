import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.algos_dir = os.path.join(data_dir, 'algoritmos')
        
    def get_availability_mask(self, end_date=None):
        """
        Scans all files in algos_dir.
        Returns a list of filenames (or IDs) that have data up to the end_date.
        If end_date is None, uses the latest date found in the dataset approx.
        """
        active_algos = []
        files = os.listdir(self.algos_dir)
        print(f"Scanning {len(files)} files for availability...")
        
        # Heuristic: Check modification time or tail of file for speed?
        # For strictness, let's read the last few lines of each file.
        # But for 14,000 files, full read is slow.
        
        count = 0
        for f in files:
            if not f.endswith('.csv'):
                continue
                
            path = os.path.join(self.algos_dir, f)
            try:
                # OPTIMIZATION: Read only the last 200 bytes to get the last line
                with open(path, 'rb') as f_obj:
                    f_obj.seek(0, 2) # Seek to end
                    size = f_obj.tell()
                    seek_size = min(size, 200)
                    f_obj.seek(-seek_size, 2)
                    tail = f_obj.read()
                    
                # Decode and split lines
                lines = tail.decode('utf-8', errors='ignore').strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    # Format: datetime, ...
                    # 2024-02-02 18:00:00,89.92,...
                    if ',' in last_line:
                        date_str = last_line.split(',')[0]
                        if date_str.startswith('2024'):
                            active_algos.append(f)
            except Exception as e:
                pass
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} files...")
                
        print(f"Found {len(active_algos)} active algorithms out of {len(files)}")
        return active_algos

    def load_tail_df(self, filename, n_lines=150):
        """
        Reads only the last n_lines of the file efficiently.
        Returns a DataFrame with 'datetime' and 'close'.
        """
        path = os.path.join(self.algos_dir, filename)
        
        try:
            with open(path, 'rb') as f_obj:
                f_obj.seek(0, 2)
                size = f_obj.tell()
                # Estimate bytes per line (approx 50-60)
                seek_size = min(size, n_lines * 80) 
                f_obj.seek(-seek_size, 2)
                tail = f_obj.read()
                
            lines = tail.decode('utf-8', errors='ignore').strip().split('\n')
            
            # We might have cut the first line partially
            if len(lines) > n_lines:
                lines = lines[-n_lines:]
            
            data = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 5:
                    # layout: datetime,open,high,low,close
                    try:
                        dt = parts[0]
                        close = float(parts[4])
                        data.append({'datetime': dt, 'close': close})
                    except:
                        pass
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception:
            return pd.DataFrame()

    def load_algo_data(self, filename):
        path = os.path.join(self.algos_dir, filename)
        return pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')

if __name__ == "__main__":
    loader = DataLoader('data')
    active = loader.get_availability_mask()
    print(f"Sample active: {active[:5]}")
