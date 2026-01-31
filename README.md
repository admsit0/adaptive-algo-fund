# Athen AI Competition (Adaptive Algo Fund)

Este repositorio contiene un sistema de **Gestión de Carteras basado en Reinforcement Learning (RL)**. Utiliza algoritmos PPO (Proximal Policy Optimization) para asignar capital dinámicamente entre un universo de activos o algoritmos de trading, con el objetivo de optimizar el ratio de Sharpe/Sortino y controlar el riesgo (Drawdown).

## Requisitos Previos

* **Python 3.10+**
* Sistema operativo: Windows, macOS o Linux.

## Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/adaptive-algo-fund.git
cd adaptive-algo-fund

```


2. **Crear un entorno virtual (Recomendado):**
*Windows:*
```bash
python -m venv .venv
.\.venv\Scripts\activate

```


*Mac/Linux:*
```bash
python3 -m venv .venv
source .venv/bin/activate

```


3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```



## Configuración de Datos

Debido a la privacidad de los datos financieros, las carpetas de datos están excluidas del control de versiones (listadas en `.gitignore`). Para que el código funcione, debes crear la estructura de directorios y añadir los archivos fuente manualmente.

### 1. Estructura de Carpetas

Asegúrate de que existan los siguientes directorios en la raíz del proyecto:

```text
data/
├── raw_algos/          <-- Ubicación de los datos de activos
├── benchmark/          <-- Ubicación del benchmark
└── processed/          <-- (Se genera automáticamente, no requiere acción)

```

### 2. Archivos Requeridos

* **Datos de Activos (`data/raw_algos/`)**:
* Coloca aquí los archivos `.csv` o `.parquet` de los activos o algoritmos que el sistema operará.
* El formato debe ser compatible con el script de ingesta (generalmente requiere columnas de fecha y equity/retorno).


* **Benchmark (`data/benchmark/`)**:
* Debes añadir el archivo `benchmark_monthly_returns.csv`.
* Este archivo es esencial para el cálculo de métricas comparativas y el entrenamiento.
* **Formato esperado:** Columnas `month` (YYYY-MM) y `monthly_return` (formato decimal).



## Ejecución del Pipeline

El sistema está diseñado para ejecutarse secuencialmente. Ejecuta los scripts ubicados en la carpeta `scripts/` en el siguiente orden:

### Paso 1: Procesamiento de Datos

Prepara los datos crudos y genera las variables (features) para el modelo.

```bash
# Ingesta y limpieza inicial
python scripts/01-ingest_data.py

# Creación de Clusters y Features (Macro + Técnicos)
python scripts/02-build_features.py

```

*Nota: Esto generará archivos `.parquet` en la carpeta `data/processed/`.*

### Paso 2: Entrenamiento (Opcional)

Entrena un agente individual en un periodo fijo definido en la configuración.

```bash
python scripts/03-train_agent.py

```

### Paso 3: Testeo / Backtest

Prueba el modelo entrenado en datos fuera de la muestra (Out-of-Sample) y genera un reporte básico de rendimiento.

```bash
python scripts/04-test_agent.py

```

### Paso 4: Walk-Forward Validation

Realiza una validación robusta utilizando ventanas rodantes (Rolling Window), reentrenando el modelo anualmente para evaluar su capacidad de adaptación y evitar el sobreajuste.

```bash
python scripts/05-walk_forward.py

```

**Salida:** Generará una carpeta en `data/processed/walk_forward_results/` conteniendo:

* Gráficos de rendimiento comparativo vs Benchmark.
* Resúmenes ejecutivos en formato `.txt` y `.csv`.
* Análisis detallados de asignación de activos y Drawdown por cada periodo (Fold).

## Configuración

Para ajustar los parámetros de la estrategia y del entorno, edita el archivo:
`src/core/config.py`

Las variables principales incluyen:

* `TRAIN_START_DATE` / `TEST_END_DATE`: Fechas para la simulación.
* `N_CLUSTERS`: Número de grupos de activos a utilizar.
* `STOP_LOSS_PCT`: Límite de gestión de riesgo duro.
* `COST_BPS`: Costes de transacción simulados (ej: 0.0005 representa 5 puntos básicos).

## Resultados

Todos los resultados, gráficos y registros de operaciones se guardan automáticamente en el directorio `data/processed/`.

* Revisa la carpeta `test_results` para los resultados de backtests simples.
* Revisa la carpeta `walk_forward_results` para los resultados de la validación completa.
