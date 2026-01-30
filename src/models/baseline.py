from sklearn.linear_model import LinearRegression
import numpy as np

class LinearBaseline:
    def __init__(self, window=60):
        self.model = LinearRegression()
        self.window = window
        
    def train(self, macro_data, cluster_data):
        """Entrena prediciendo retornos futuros de clusters con datos macro."""
        # X: Macro T-1, Y: Cluster T
        X = macro_data.shift(1).dropna()
        Y = cluster_data.loc[X.index]
        self.model.fit(X, Y)
        
    def predict(self, current_macro):
        """Retorna vector de pesos sugeridos (Heurística simple)."""
        # Predicción de retornos para mañana
        pred_rets = self.model.predict(current_macro.reshape(1, -1))[0]
        
        # Estrategia: Invertir solo en clusters positivos, proporcional a su predicción
        weights = np.maximum(pred_rets, 0)
        if weights.sum() == 0:
            return np.zeros(len(weights) + 1) # Todo a Cash si nada pinta bien
            
        # Normalizar
        weights = weights / weights.sum()
        # Añadir cash (0) al final
        return np.append(weights, 0)
    