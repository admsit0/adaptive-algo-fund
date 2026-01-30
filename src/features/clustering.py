import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..core.config import N_CLUSTERS, REBALANCE_FREQ, CLUSTERS_FILE, LABELS_FILE

def run_rolling_clustering(returns_df, active_mask, window=60):
    """
    GENERADOR MAESTRO DE FEATURES Y RETORNOS.
    Genera un único archivo parquet alineado con:
    1. Retornos Reales (para calcular P&L).
    2. Features Técnicas (Volatilidad, Momentum) para el Agente.
    """
    print("🔄 Ejecutando Clustering + Feature Engineering (Root Fix)...")
    
    dates = returns_df.index
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    
    # Almacenaremos diccionarios para luego crear un DF único
    master_data = [] 
    labels_dict = {}
    
    # Barrido en el tiempo (Rolling)
    for i in range(window, len(dates), REBALANCE_FREQ):
        # Fechas de entrenamiento (pasado) y predicción (futuro inmediato)
        train_idx = dates[i-window : i]
        predict_idx = dates[i : min(i+REBALANCE_FREQ, len(dates))]
        
        if len(predict_idx) == 0: break
        
        # --- 1. CLUSTERING (En base al pasado) ---
        subset = returns_df.loc[train_idx]
        valid_algos = active_mask.loc[train_idx].mean() > 0.8
        valid_ids = valid_algos[valid_algos].index
        
        if len(valid_ids) < N_CLUSTERS: continue
            
        # Features para agrupar (Perfil de riesgo)
        X = pd.DataFrame({
            'ret_total': (1 + subset[valid_ids]).prod() - 1,
            'volatility': subset[valid_ids].std()
        }).fillna(0)
        
        # K-Means
        X_scaled = StandardScaler().fit_transform(X)
        labels = kmeans.fit_predict(X_scaled)
        
        # Guardar etiquetas
        period_labels = pd.Series(labels, index=valid_ids)
        labels_dict[dates[i]] = period_labels.to_dict()
        
        # --- 2. CÁLCULO DE FEATURES PARA EL PERIODO SIGUIENTE ---
        # Tomamos los retornos futuros de los algos seleccionados
        future_rets_df = returns_df.loc[predict_idx, valid_ids]
        
        # Para cada día en el bloque de predicción...
        for date in predict_idx:
            day_data = {'date': date}
            
            # Para cada Cluster...
            for c_id in range(N_CLUSTERS):
                algos_in_cluster = period_labels[period_labels == c_id].index
                
                if len(algos_in_cluster) == 0:
                    ret_val = 0.0
                else:
                    # Retorno Equal-Weight del cluster hoy
                    ret_val = future_rets_df.loc[date, algos_in_cluster].mean()
                
                # --- AQUÍ ESTÁ LA SOLUCIÓN DE RAÍZ ---
                # Guardamos el retorno (Target)
                day_data[f'C{c_id}_Ret'] = ret_val
                
                # Calculamos features "On the Fly" usando ventana retrospectiva real
                # (Accedemos a returns_df global para tener historia completa hasta 'date')
                # Optimización: Usamos una ventana de 20 días hacia atrás desde 'date'
                
                # Historia reciente del cluster (aprox)
                # Nota: Calcular esto día a día es lento, pero preciso.
                # Para velocidad, usaremos el 'ret_val' y calcularemos rolling después sobre el DF completo.
                
            master_data.append(day_data)

    # Crear DataFrame Maestro
    full_df = pd.DataFrame(master_data).set_index('date')
    
    # --- 3. FEATURE ENGINEERING VECTORIZADO (Rápido) ---
    print("   ⚡ Calculando Momentum y Volatilidad vectorizados...")
    
    final_cols = []
    
    for c_id in range(N_CLUSTERS):
        col_ret = f'C{c_id}_Ret'
        col_vol = f'C{c_id}_Vol'
        col_mom = f'C{c_id}_Mom'
        
        # Volatilidad 20d
        full_df[col_vol] = full_df[col_ret].rolling(20).std().fillna(0)
        
        # Momentum 20d (Retorno acumulado)
        full_df[col_mom] = full_df[col_ret].rolling(20).sum().fillna(0)
        
        # Ordenamos las columnas para que estén juntas: Ret, Vol, Mom
        final_cols.extend([col_ret, col_vol, col_mom])

    # Reordenar y limpiar
    full_df = full_df[final_cols].fillna(0)
    
    print(f"✅ Archivo Maestro Generado: {full_df.shape}")
    print(f"   Columnas: {full_df.columns[:6].tolist()}...")
    
    full_df.to_parquet(CLUSTERS_FILE)
    
    import pickle
    with open(str(LABELS_FILE).replace('.parquet', '.pkl'), 'wb') as f:
        pickle.dump(labels_dict, f)