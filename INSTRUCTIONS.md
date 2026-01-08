### 1. Fase de Ingeniería Inversa (El "Benchmark")

La web menciona explícitamente que *debes* deducir el capital y las reglas del benchmark. No empieces a programar tu modelo hasta entender contra qué compites.

* **Objetivo:** Determinar el AUM (Assets Under Management) latente.
* **Qué analizar:**
  * Mira el tamaño de las posiciones abiertas simultáneamente en el histórico del benchmark.
  * Calcula el *drawdown* máximo que tolera.
  * **Hipótesis a probar:** ¿Es un sistema de "Reversión a la media" o de "Seguimiento de tendencia"? ¿Opera todos los días o solo en ciertos regímenes?

### 2. Gestión de Datos: El "Filtro de Supervivencia"

La web advierte repetidamente sobre el  **Sesgo de Supervivencia** . Esta es la trampa principal.

* **Estrategia:** Crea una "Máscara de Disponibilidad".
  * Antes de cualquier cálculo, tu código debe filtrar Algo_i si Fecha_actual < Fecha_inicio_Algo o Fecha_actual > Fecha_fin_Algo.
  * **Prueba:** Si entrenas con datos futuros (algoritmos que sabes que sobreviven 4 años), tu modelo tendrá un *overfitting* masivo y fallará en la validación ciega.

### 3. Arquitectura del Modelo (El Meta-Algoritmo)

Dado que tienes 14.000 activos (algoritmos), el problema principal es la  **reducción de dimensionalidad** . No puedes optimizar una cartera de 14.000 activos eficientemente en menos de 10 minutos de ejecución.

Enfoque A: El "Investment Clock" (Recomendado por AthenAI)

La web sugiere inferir qué hacen los algoritmos (Oro, SP500, Renta Fija).

* **Paso 1 (Clusterización):** Usa aprendizaje no supervisado (K-Means o DBSCAN) para agrupar los 14.000 algoritmos en, digamos, 50 "super-familias" basadas en su correlación.
* **Paso 2 (Rotación):** Identifica en qué "hora" del reloj económico estamos basándote en qué clusters lo están haciendo bien (ej. si los clusters de "commodities" suben, quizás hay inflación).
* **Paso 3 (Selección):** Invierte en el top 5% de algoritmos del cluster que mejor se adapte al régimen actual.

Enfoque B: Aprendizaje por Refuerzo (RL)

Mencionado como "recomendado" pero no obligatorio.

* **Estado (State):** Rendimiento reciente de los top 100 algos + Volatilidad del mercado.
* **Acción (Action):** Asignar pesos (ponderación) a una sub-cesta de algoritmos.
* **Recompensa (Reward):** Ratio de Sharpe o Sortino (para penalizar la volatilidad, no solo buscar retorno).

### 4. Qué probar (Tu "Hoja de Ruta")

Te sugiero iterar en este orden para no perderte:

1. **Semana 1: Baseline Simple (Ranking).**
   * Ignora ML complejo. Simplemente selecciona cada mes los 20 algoritmos con mejor *Momentum* (rendimiento) de los últimos 3 meses, filtrando por baja volatilidad.
   * *Objetivo:* Batir al benchmark con reglas simples. Si esto funciona, ya tienes una base sólida.
2. **Semana 2: Feature Engineering.**
   * Crea métricas para cada algoritmo: *Beta* (correlación con el promedio), *Max Drawdown* histórico, *Consistencia* (días ganadores vs perdedores).
3. **Semana 3: Modelo Predictivo (XGBoost/LightGBM).**
   * Entrena un modelo para predecir la probabilidad de que un algoritmo tenga retorno positivo la semana siguiente.
   * *Ventaja:* Los modelos basados en árboles son muy rápidos en inferencia (crucial para el límite de 10 min).
4. **Semana 4: Asignación de Capital (Risk Parity).**
   * No asignes el mismo dinero a todos. Dale menos dinero a los algoritmos volátiles y más a los estables para equilibrar el riesgo global del portafolio.

### 5. Restricciones Técnicas (Crucial)

* **Tiempo de ejecución < 10 min:** Evita redes neuronales profundas (Transformers pesados) que tarden mucho en inferencia sin GPU potente. Los modelos pre-entrenados deben ser ligeros.
* **Sin Internet:** Todo debe estar contenido en tu entrega. No puedes llamar a Yahoo Finance ni a OpenAI. Si necesitas datos externos, deben estar implícitos en el comportamiento de los 14.000 algoritmos.

**Resumen del primer paso:** Descarga los datos, visualiza la curva de equidad del benchmark y averigua cuánto dinero está moviendo. Sin ese número, no podrás dimensionar tus apuestas.
