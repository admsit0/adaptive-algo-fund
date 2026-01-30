import pandas as pd

def distribute_capital(total_capital, cluster_weights, current_date, labels_dict, active_mask_df=None):
    """
    Versión robusta que verifica si el algo sigue vivo HOY.
    """
    # 1. Recuperar la composición del último rebalanceo
    valid_dates = [d for d in labels_dict.keys() if d <= current_date]
    if not valid_dates: return {}
    
    latest_rebal = max(valid_dates)
    mapping = labels_dict[latest_rebal] # {Algo1: 0, Algo2: 1...}
    
    allocation = {}
    
    # 2. Agrupar algos por cluster
    clusters = {}
    for algo, c_id in mapping.items():
        clusters.setdefault(c_id, []).append(algo)
        
    # 3. Repartir dinero
    for c_id, weight in enumerate(cluster_weights):
        if weight <= 0: continue
        
        # Candidatos originales del cluster
        candidates = clusters.get(c_id, [])
        
        # --- FIX: FILTRO DE SUPERVIVENCIA DIARIA ---
        # Si pasamos active_mask, filtramos los que estén muertos HOY
        if active_mask_df is not None:
            # Asumimos que active_mask tiene fechas en índice y columnas=algos
            # Verificamos si existe la fecha y el algo es True/1
            if current_date in active_mask_df.index:
                alive_today = active_mask_df.loc[current_date]
                # Filtramos: Solo los que están en candidates Y están activos hoy
                candidates = [algo for algo in candidates if alive_today.get(algo, False)]
        
        if not candidates: continue
            
        # Repartimos solo entre los supervivientes de hoy
        budget_cluster = total_capital * weight
        amount_per_algo = budget_cluster / len(candidates)
        
        for algo in candidates:
            allocation[algo] = amount_per_algo
            
    return allocation
