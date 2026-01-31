Transcripción del diagrama manuscrito:

**1. Selección de Activos**

* **14k ALGOS (ACCIONES)**
* Cómo va a tradear? Cuales son los outputs? Evaluar bases.md y decidir
* Poner stop loss, leverage y take profit como valores discretos (pips) SI LAS BASES lo permiten, claro


* **5k ALGOS VÁLIDOS POR FECHA**

**2. Procesamiento de Datos (Reducción)**

* **REDUCIR A CLUSTERING DINÁMICO**
* (N CLUSTERS SOBRE LOS T PERIODOS ANTERIORES CORRELANDO INDICADORES, MOMENTUM, ETC)
* *[Texto en verde]* ELEGIR INDICADORES  NOISE filtering podría ser útil?
* *[Texto en rojo]* Imprimirlos con su catacterización



**3. Modelo de Aprendizaje (RL)**

* 
* **Feed al modelo de RL (OpenAI PPO)**
* Los N clusters con su caracterización completa **EN CADA FECHA**



**4. Ejecución / Decisión**
 
* **Debe elegir si asumirá ETF ponderado (criterio?)**
* Se le pasarán también variables macroeconómicas (sacadas de un estimador (modelo clásico?) porque no hay llamadas externas) Eg: fear index

---

**NOTA (Cuadro superior derecha):**

* Comparar con HOLD (Invertir en todos los algos a partes iguales)  -> MKT
* Comparar con linear regression

**TESTS (Abajo izquierda):**

* In sample excellence
* Validation excellence with no overfitting
* Permutation tests (to evaluate paths)
* *[Llave lateral]* Cross-val (Train 5 times leaving out one year)
