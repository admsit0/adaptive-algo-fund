"""
RL pipeline wrapper que asegura que el estado RL incluye variables macro relevantes:
- En entrenamiento: añade variables macro reales.
- En predicción: añade proxies/estimadas.
"""
import polars as pl
import numpy as np

def enrich_rl_state(cluster_features: pl.DataFrame, macro_features: pl.DataFrame = None):
    """
    Une el estado RL (cluster_features) con variables macro (macro_features) por fecha.
    Si macro_features es None, devuelve cluster_features tal cual.
    """
    if macro_features is not None:
        if 'date' in cluster_features.columns and 'date' in macro_features.columns:
            enriched = cluster_features.join(macro_features, on='date', how='left')
        else:
            # Si no hay fecha, añadir broadcast
            for col in macro_features.columns:
                if col != 'date':
                    cluster_features = cluster_features.with_columns([
                        pl.lit(macro_features[col][0]).alias(col)
                    ])
            enriched = cluster_features
        return enriched
    else:
        return cluster_features

# Ejemplo de uso en pipeline RL:
# state = enrich_rl_state(cluster_features, macro_features)
# agent.select_action(state)
