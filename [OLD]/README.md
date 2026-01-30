# AthenAI Competition - Macro Portfolio Management

## Objetivo

Sistema de gestion de carteras que toma decisiones de inversion sobre ~14,000 algoritmos de trading, superando al benchmark propuesto. El modelo final funciona **completamente offline** sin acceso a APIs ni datos externos.

---

## Resultados Actuales (Phase 2)

### Proxy Models Performance

| Proxy | Tarea | Metrica Principal | Resultado | Estado |
|-------|-------|-------------------|-----------|--------|
| **VIX Spike** | Clasificacion (delta log > q90) | AUC | 0.50 +/- 0.10 | Baseline |
| **Rates** | Regresion categorica | Sign Accuracy | **80.6%** +/- 10.8% | Bueno |
| **Risk-Off** | Clasificacion future window | AUC | 0.46 +/- 0.10 | Mejorable |
| **Factors** | Binario MOM > 0 | - | 54 muestras | Datos limitados |

### Validaciones de Calidad

| Test | Proxy | Resultado |
|------|-------|-----------|
| **Leakage Test** | Rates | PASADO (real=0.068 vs shuffle=0.028) |
| **Feature Independence** | Risk-Off | PASADO (todos r < 0.9) |
| **Drift Monitoring** | Rates | +87% drift detectado (mejora en tiempo) |
| **Calibration** | VIX, Risk-Off | ECE reducido con Platt scaling |

---

## Arquitectura del Pipeline

### Flujo de Datos

    PIPELINE COMPLETO
    ==================
    
    +-------------+    +-------------+    +-------------------------+
    |   Paso 0    |    |   Paso 1    |    |        Paso 2           |
    | Preprocess  |--->| Personality |--->|      Clustering         |
    |   14,000    |    |   Features  |    |   100 super-activos     |
    |  algoritmos |    |  estaticas  |    |                         |
    +-------------+    +-------------+    +-----------+-------------+
                                                      |
                                                      v
                        +----------------------------------------+
                        |        Paso 3 - PROXY LAYERS           |
                        |                                        |
    +-----------------+ | +--------+ +--------+ +--------+       |
    | External Data   | | |Proxy A | |Proxy B | |Proxy C |       |
    | (TRAIN only)    | | |  VIX   | | Rates  | |Risk-Off|       |
    | - VIXCLS        | | | Spike  | |Category| | Future |       |
    | - DGS10         | | |Detector| |Predict | | Window |       |
    | - FF Factors    | | +--------+ +--------+ +--------+       |
    +-----------------+ |                |                       |
                        |                v                       |
                        |    +------------------------+          |
                        |    |   Enriched RL State    |          |
                        |    | cluster_features+proxy |          |
                        |    | predictions (35 cols)  |          |
                        |    +------------------------+          |
                        +----------------------------------------+
                                         |
                                         v
                        +----------------------------------------+
                        |         Paso 4 - RL AGENT              |
                        |   PPO/DQN sobre estado enriquecido     |
                        |   (En desarrollo)                      |
                        +----------------------------------------+

---

## Modelos y Tecnicas Implementadas

### Paso 3: Proxy Layers (COMPLETADO)

#### Proxy A: VIX Synthetic
**Objetivo**: Detectar spikes de volatilidad sin acceso al VIX real.

Configuracion actual (layers.yaml):
- target_transform: spike (Clasificacion binaria)
- use_spike_detection: true
- spike_quantile: 0.90 (Top 10% = spike)
- use_multi_horizon: true
- horizons: [1, 5] (t+1 fast, t+5 macro)
- use_stress_features: true

**Features mas importantes**:
1. avg_vol_20 (0.136)
2. f_vix_log_current (0.134)
3. realized_vol_mkt (0.105)
4. f_pca_1 (0.088)

**Calibracion**: Platt scaling reduce ECE de 0.036 -> 0.0007

#### Proxy B: Rate Oracle
**Objetivo**: Predecir direccion de tipos de interes (DGS10).

Configuracion:
- horizon: 10 (t+10 dias)
- use_categorical: true (down/flat/up)
- up_threshold_bps: 5.0
- run_leakage_test: true (PASADO)
- monitor_drift: true (+87% drift)

**Features mas importantes**:
1. jumpiness_mkt (0.262)
2. realized_vol_mkt (0.235)
3. f_mkt_cluster (0.225)
4. tail_q95_cs (0.118)

**Uncertainty**: residual_std = 0.66 +/- 0.07

#### Proxy C: Risk-Off Detector
**Objetivo**: Predecir drawdowns futuros (no pasados).

Configuracion:
- mode: internal (Sin datos externos)
- use_future_window: true (CRITICO!)
- future_horizon: 10 (Predict next 10 days)
- use_quantile_threshold: true
- risk_quantile: 0.10 (Bottom 10% = risk-off)
- verify_feature_independence: true (PASADO)

**Features mas importantes**:
1. realized_vol_mkt (0.126)
2. avg_vol_20 (0.123)
3. dispersion_ret_cs (0.092)
4. stress_dd_min_60 (0.076)

#### Proxy D: Factor Monitor
**Objetivo**: Predecir si MOM > 0 (momentum positivo).

Configuracion:
- use_binary_mode: true
- binary_target: mom
- 54 muestras mensuales, sin CV suficiente

### Universe Features (26 features)

- f_mkt_cluster: Market return proxy
- breadth_pos_ret: % clusters positive
- dispersion_ret_cs: Cross-sectional std
- avg_vol_20: Mean volatility
- avg_corr_20, avg_corr_60: Correlation regime
- tail_q05_cs, tail_q95_cs: Tail risk
- stress_dd_min_60: Worst drawdown
- realized_vol_mkt: sigma(f_mkt_cluster)
- jumpiness_mkt: max|ret| in window
- corr_spike: corr_20 - corr_60
- skew_cs: Cross-sectional skew
- lowvol_highvol_spread: Factor spread
- momentum_spread: Factor spread
- f_pca_1..f_pca_10: PCA factors

---

## Estructura del Proyecto

    competicion AthenAI/
    |-- pyproject.toml              # Configuracion del paquete
    |-- README.md                   # Este archivo
    |-- STRATEGY.md                 # Documento de estrategia
    |
    |-- configs/
    |   |-- preprocess.yaml         # Paso 0
    |   |-- personality.yaml        # Paso 1
    |   |-- clustering.yaml         # Paso 2
    |   +-- layers.yaml             # Paso 3 (NUEVO)
    |
    |-- data/
    |   |-- datos_competicion/      # ~14,000 CSVs de algoritmos
    |   |-- cache/
    |   |   |-- clustering_v1/      # Outputs de clustering
    |   |   +-- layers_v3/          # Outputs de proxy layers
    |   |       |-- universe_features_daily_*.parquet
    |   |       |-- proxy_preds_*.parquet
    |   |       +-- cluster_features_enriched_*.parquet
    |   +-- reports/
    |       +-- proxy_training_*.md
    |
    |-- src/athenai/
    |   |-- core/                   # Utilidades compartidas
    |   |   |-- artifacts.py
    |   |   |-- config.py
    |   |   +-- logging.py
    |   |
    |   |-- data/                   # Paso 0: Preprocesamiento
    |   |   |-- algos_panel.py
    |   |   +-- algos_features.py
    |   |
    |   |-- features/               # Paso 1: Personalidad
    |   |   +-- personality.py
    |   |
    |   |-- clustering/             # Paso 2: Clustering
    |   |   |-- build_clusters.py
    |   |   +-- cluster_timeseries.py
    |   |
    |   |-- layers/                 # Paso 3: Proxy Layers (NUEVO)
    |   |   |-- config.py           # ProxyVIXConfig, ProxyRatesConfig, etc.
    |   |   +-- proxies/
    |   |       |-- base.py         # ProxyTask, ProxyTrainResult
    |   |       |-- datasets.py     # ProxyDatasetBuilder
    |   |       |-- trainer.py      # ProxyTrainer + leakage/drift
    |   |       +-- models_linear.py # Ridge, Logistic, Softmax
    |   |
    |   |-- pipelines/              # Orquestadores
    |   |   |-- preprocess.py
    |   |   |-- personality.py
    |   |   |-- clustering.py
    |   |   +-- layers.py           # LayersPipeline (NUEVO)
    |   |
    |   |-- external/               # Datos externos (NUEVO)
    |   |   +-- loaders.py          # FRED API, FF data
    |   |
    |   +-- scripts/                # Entry points CLI
    |       |-- run_preprocess.py
    |       |-- run_personality.py
    |       |-- run_clustering.py
    |       +-- train_layers.py     # (NUEVO)
    |
    |-- notebooks/
    |   +-- 00_exploracion_datos.ipynb
    |
    +-- tests/
        |-- test_preprocess_smoke.py
        |-- test_personality_smoke.py
        +-- test_clustering_smoke.py

---

## Guia de Ejecucion

### Instalacion

    pip install -e .

### Pipeline Completo

    # Paso 0: Preprocesamiento
    python -m athenai.scripts.run_preprocess --config configs/preprocess.yaml --overwrite
    
    # Paso 1: Features de personalidad
    python -m athenai.scripts.run_personality --latest --overwrite
    
    # Paso 2: Clustering (actualizar run_ids en clustering.yaml primero)
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml --overwrite
    
    # Paso 3: Proxy Layers (NUEVO)
    python -m athenai.scripts.train_layers --config configs/layers.yaml --train

### Solo Prediccion (Inference Mode)

    # Usa modelos entrenados, sin datos externos
    python -m athenai.scripts.train_layers --config configs/layers.yaml --predict

---

## Outputs del Pipeline

### Archivos Generados (layers_v3/)

| Archivo | Descripcion | Shape |
|---------|-------------|-------|
| universe_features_daily_*.parquet | 26 features agregadas por fecha | (1430, 27) |
| proxy_preds_*.parquet | Predicciones de todos los proxies | (1430, 11) |
| cluster_features_enriched_*.parquet | **Estado para RL** | (106236, 35) |
| external_*.parquet | Datos externos cacheados | Variable |

### Columnas del Estado Enriquecido (35 total)

Identificadores:
- date, cluster_id, algo_count, is_alive

Features de cluster (21):
- ret_ew, vol_ew_20, momentum_ew_20, sharpe_ew_20
- max_dd_ew_60, avg_corr_intra, avg_beta_mkt, ...

Predicciones de proxies (10):
- pred_vix_synthetic: log(VIX) predicho
- pred_vix_spike: P(spike) - binario
- pred_rate_change: delta bps predicho
- pred_risk_off: P(risk_off) - binario
- pred_factor_smb: P(SMB wins)
- pred_factor_hml: P(HML wins)
- pred_factor_mom: P(MOM wins)
- pred_vix_uncertainty: sigma(residual) del proxy
- pred_rate_uncertainty
- pred_vix_spike_h5: Spike a t+5

---

## Ideas y Mejoras Pendientes

### Corto Plazo
- [ ] Mejorar VIX AUC con mas stress features (corr_spike, implied vol proxy)
- [ ] Agregar autoregressive feature para VIX (pred_t-1 -> pred_t)
- [ ] Investigar drift en Rates (cambio de regimen 2020-2024?)

### Medio Plazo
- [ ] Implementar Investment Clock (4 fases macro)
- [ ] Entorno de RL con estado enriquecido
- [ ] Reward shaping para Calmar Ratio

### Largo Plazo
- [ ] Ensemble de proxies con uncertainty weighting
- [ ] Meta-learning para adaptacion rapida
- [ ] Backtesting con transaction costs

---

## Metricas de Evaluacion

### Para Proxies
- **Clasificacion**: AUC, Brier Score, ECE (calibracion)
- **Regresion**: R2, Sign Accuracy, Spearman correlation

### Para RL Agent (futuro)
- **Calmar Ratio** = CAGR / Max Drawdown (objetivo principal)
- **Sharpe Ratio** anualizado
- **Max Drawdown** < 20%
- **Win Rate** mensual > 55%

---

## Configuracion Avanzada

### Variables de Entorno

    # API de FRED (opcional, usa mock data si no esta)
    set FRED_API_KEY=your_api_key_here

### Walk-Forward CV

    cv_config:
      n_folds: 5
      val_months: 6
      min_train_months: 12
      gap_days: 1              # Evitar leakage
      enable_calibration: true  # Platt scaling
      compute_uncertainty: true # Residual std

---

## Changelog

### v3 (2026-01-20) - Phase 2
- VIX cambiado a spike detection (clasificacion)
- Multi-horizonte t+1, t+5
- Leakage test para Rates
- Drift monitoring
- Risk-off con future window
- Feature independence check
- Quantile-based threshold dinamico

### v2 (2026-01-19) - Phase 1
- Proxy Layers pipeline completo
- VIX delta log target
- Rates categorical (down/flat/up)
- Recession internal mode
- Factors binary MOM > 0
- Platt calibration + ECE
- Uncertainty estimation

### v1 (2026-01-19) - Initial
- Preprocesamiento 14,000 algoritmos
- Features de personalidad
- Clustering 100 super-activos

---

## El Problema Original

La competicion presenta un desafio particular: el modelo final debe funcionar completamente offline, sin acceso a APIs ni datos externos. Esto significa que no podemos consultar indicadores macroeconomicos tradicionales (PIB, IPC, tipos de interes) durante la ejecucion.

## La Solucion

La arquitectura se basa en capas secuenciales:

1. **Capa 0 - Preprocesamiento**: Transformar los CSVs crudos en datos estructurados y limpios
2. **Capa 1 - Features y Clustering**: Extraer la personalidad de cada algoritmo y agruparlos en super-activos
3. **Capa 2 - Proxy Layers**: Inferir indicadores macro usando SOLO datos internos del mercado
4. **Capa 3 - Agente RL**: Tomar decisiones de inversion basadas en el estado enriquecido

---

## Requisitos Previos

### Instalar el Paquete

El proyecto usa pyproject.toml para la configuracion. Instala en modo desarrollo:

    pip install -e .

### Estructura de Datos Esperada

Los datos de la competicion deben estar en data/datos_competicion/:

    data/
      datos_competicion/
        algoritmos/           # ~14,000 CSVs, uno por algoritmo
          00eyz.csv
          00kZg.csv
          ...
        macro_data.csv        # Datos macroeconomicos historicos
        benchmark_monthly_returns.csv
        benchmark_yearly_returns.csv
        trades_benchmark.csv

---

## Documentacion Detallada por Paso

### Paso 0: Preprocesamiento

Objetivo: Transformar los CSVs crudos en datos estructurados y limpios.

Entrada:
- ~14,000 archivos CSV en data/datos_competicion/algoritmos/

Salida:
- trades_panel.parquet: Panel de trades normalizado
- manifest.json: Metadatos de la ejecucion

Comando:

    python -m athenai.scripts.run_preprocess --config configs/preprocess.yaml --overwrite

### Paso 1: Features de Personalidad

Objetivo: Extraer caracteristicas estaticas que describen el comportamiento de cada algoritmo.

Entrada:
- trades_panel.parquet del Paso 0

Salida:
- personality_features.parquet: Features por algoritmo
- report_personality_*.md: Reporte de analisis

Features extraidas:
- return_mean, return_std: Estadisticas basicas
- sharpe_annualized: Ratio riesgo/retorno
- sortino_annualized: Penaliza solo downside
- max_drawdown: Peor caida historica
- vol_of_vol: Estabilidad de la volatilidad
- hit_rate: Porcentaje de dias positivos
- avg_trade_duration: Duracion media de operaciones
- trades_per_month: Frecuencia operativa

Comando:

    python -m athenai.scripts.run_personality --latest --overwrite

### Paso 2: Clustering

Objetivo: Agrupar algoritmos similares en 100 super-activos para reducir dimensionalidad.

Entrada:
- personality_features.parquet del Paso 1
- trades_panel.parquet del Paso 0

Salida:
- cluster_assignments.parquet: Mapeo algoritmo -> cluster
- cluster_timeseries.parquet: Series temporales agregadas
- pca_model.npz: Modelo PCA para transformaciones

Comando:

    python -m athenai.scripts.run_clustering --config configs/clustering.yaml --overwrite

### Paso 3: Proxy Layers (NUEVO - Phase 2)

Objetivo: Entrenar modelos que predicen indicadores externos usando solo datos internos.

Entrada:
- cluster_timeseries.parquet del Paso 2
- Datos externos (SOLO durante entrenamiento): VIX, DGS10, Fama-French

Salida:
- universe_features_daily.parquet: Features agregadas del universo
- proxy_preds.parquet: Predicciones de todos los proxies
- cluster_features_enriched.parquet: Estado completo para RL
- proxy_training_report.md: Metricas y analisis

Comando:

    python -m athenai.scripts.train_layers --config configs/layers.yaml --train

---

## Proxies: Detalle Tecnico

### VIX Synthetic (Spike Detection)

Transformacion del target:
1. Raw VIX -> log(VIX) para estabilizar varianza
2. delta_log = log(VIX_t+h) - log(VIX_t)
3. spike = 1 si delta_log > quantile(0.90) else 0

Walk-Forward CV:
- 5 folds con 6 meses de validacion
- Gap de 1 dia entre train/val
- Calibracion Platt para probabilidades

### Rate Oracle (Categorical)

Transformacion del target:
- down: delta_bps < -5
- flat: -5 <= delta_bps <= +5
- up: delta_bps > +5

Validaciones:
- Leakage test: Compara R2 real vs features shuffleadas
- Drift monitoring: Mide cambio de R2 entre primeros y ultimos folds

### Risk-Off Detector (Future Window)

Target construction:
1. future_ret = mean(ret[t+1:t+h]) donde h=10
2. threshold = quantile(future_ret, 0.10)
3. risk_off = 1 si future_ret < threshold else 0

Validaciones:
- Feature independence: |corr(X, y)| < 0.9 para todas las features
- Verifica que no hay leakage temporal

---

## Proyecto para la competicion AthenAI 2025-2026
