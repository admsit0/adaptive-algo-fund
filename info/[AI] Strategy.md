# Adaptive Algo Fund: Estrategia de Gestión Dinámica de Carteras (RL-Based Fund of Funds)

## 1. Tesis de Inversión
El mercado de algoritmos individuales es ruidoso y estocástico (alta tasa de fallo). Sin embargo, los **regímenes de mercado** (Tendencia, Rango, Volatilidad) son persistentes. 
Nuestra estrategia no selecciona "el mejor algoritmo", sino que construye una cartera dinámica de **"Super-Activos" (Clusters)** gestionada por un agente de Reinforcement Learning (PPO) que adapta su exposición según el régimen de mercado detectado.

---

## 2. Arquitectura de Datos (The Pipeline)

### Nivel 1: Ingestión & Universo (Raw Layer)
* **Input:** 14.000 series temporales de algoritmos (Daily/Hourly equity curves).
* **Filtrado:** `ActiveMask`. Solo algoritmos vivos en $T$.

### Nivel 2: Reducción Dimensional (Clustering Layer)
* **Objetivo:** Eliminar riesgo idiosincrático y reducir el espacio de estados.
* **Método:** K-Means con $k=10$.
* **Features de Entrada al K-Means:** Sharpe Ratio (60d), Volatilidad (60d), Correlación media.
* **Output:** 10 Series temporales sintéticas (`C0`...`C9`) que representan el retorno promedio *equal-weight* de los algos en cada cluster.

### Nivel 3: Feature Engineering (State Space)
El Agente es "ciego" a los precios. Necesita ver la derivada del precio (comportamiento). Para cada Cluster $i$, calculamos 3 vectores en ventana rodante:
1.  **Retorno ($r_t$):** $\frac{P_t}{P_{t-1}} - 1$. (Input para el Bolsillo).
2.  **Volatilidad ($\sigma_t$):** Desviación estándar de 20 días. (Input para los Ojos).
3.  **Momentum ($M_t$):** Suma de retornos de 20 días. (Input para los Ojos).

**Matriz de Estado ($S_t$):** Dimensiones $[WindowSize \times (N_{clusters} \times 3 + N_{macro})]$.

---

## 3. El Agente (The Brain)

### Modelo: PPO (Proximal Policy Optimization)
* **Actor-Critic:** Una red decide los pesos (Actor) y otra estima cuánto ganará (Critic).
* **Arquitectura:** Red Densa (MLP) `[128, 128]` con activación `Tanh`.

### Acción ($A_t$)
* Vector continuo de $N+1$ elementos (10 Clusters + 1 Cash).
* Salida: `Softmax` (Garantiza $\sum w_i = 1$).
* **Allocation:** El peso asignado a $C_i$ se distribuye equitativamente entre los algos que componen $C_i$ ese día.

### Función de Recompensa (Reward Function)
Objetivo: Maximizar Alpha sobre Benchmark minimizando Drawdown.

$$R_t = (\alpha_t \times 10) - \text{Penalty}_{DD} - \text{Penalty}_{Turnover}$$

Donde:
* $\alpha_t = R_{portfolio} - R_{benchmark}$
* $\text{Penalty}_{DD} = 50 \times |R_{port}|$ si $R_{port} < -1.5\%$ (Castigo al dolor).
* $\text{Penalty}_{Turnover} = \text{Coste de cambio de posición}$.

---

## 4. Ejecución y Validación
* **Split:** 80% Train (2020-2023), 20% Test (OOS 2024).
* **Benchmark:** Índice de referencia corregido (Base 100), ignorando flujos de caja externos.
* **Costes:** 5 bps (0.05%) por transacción simulada.
