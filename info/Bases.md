# 📊 **Bases Técnicas — Competición Finanzas y Mercados Financieros (AthenAI)**

## 🎯 **Objetivo de la Competición**

* El reto consiste en **construir un “algoritmo de algoritmos de inversión”** capaz de seleccionar entre un universo de más de **14.000 algoritmos de inversión** el conjunto o criterio que mejor *supere al benchmark proporcionado*. ([athenai.institute][1])
* A cada participante se le proporciona un **conjunto de datos (training set)** para diseñar el algoritmo; después su rendimiento será evaluado en un **set de validación al que no se tiene acceso** durante la fase de diseño. ([athenai.institute][1])

## 🧠 **Datos y Consideraciones Específicas**

### 📈 **Naturaleza de los datos**

* Los datos proporcionados **son completamente reales** —no son sintéticos— por lo que **no está permitido alterarlos ni rellenar datos faltantes**. ([athenai.institute][1])
* Debido a que muchos algoritmos “nacen y mueren” en distintos momentos del periodo (2020–2024), **solo se puede invertir en algoritmos activos en la fecha correspondiente** (i.e., hay sesgo de supervivencia y fechas de actividad distintas). ([athenai.institute][1])

### ⚖️ **Sesgo de supervivencia**

* El benchmark y los algoritmos tienen diferentes fechas de inicio y final, por lo que **no se puede invertir en un algoritmo que aún no exista o que ya haya dejado de existir en la fecha de simulación**. ([athenai.institute][1])

---

## 📊 **Reglas de Construcción y Benchmark**

### 🔍 **Inferencia del benchmark**

* La organización proporciona **solo las operaciones del benchmark**, sin detallar cómo fue construido, por lo que la **primera tarea es deducir sus reglas**: ([athenai.institute][1])

  * **Capital gestionado** (a inferir a partir de volumen y tamaño de operaciones). ([athenai.institute][1])
  * **Tamaño promedio de las operaciones**. ([athenai.institute][1])
  * **Frecuencia de negociación y estilo de inversión** (long/short, intradía, etc.). ([athenai.institute][1])

### 📌 **Alineamiento con el estilo del benchmark**

* Aunque se puede usar todo el universo de algoritmos disponibles, **el algoritmo que construyas debe seguir el estilo de inversión observado en el benchmark** (por ejemplo, el capital y frecuencia típicos). ([athenai.institute][1])
* Ejemplo: si inferimos que el benchmark invierte alrededor de 10 M €, **no puedes suponer capital infinito** en tu diseño. ([athenai.institute][1])

---

## 🧠 **Composición de los Algoritmos**

* No se trabaja con activos financieros tradicionales (acciones, bonos…) sino **con algoritmos de inversión como unidades de decisión**. ([athenai.institute][1])
* La composición interna real de los 14.000 algoritmos **no se revela** (qué activos negocian, reglas internas, etc.), aunque sí se sabe que: ([athenai.institute][1])

  * Están **auditados**. ([athenai.institute][1])
  * Disponen de **sistemas de control de riesgo** (pero su estilo/riesgo puede variar). ([athenai.institute][1])
* Se sugiere (como recomendación opcional) aplicar técnicas como **Investment Clock o análisis de ciclo de mercado** para entender mejor el comportamiento relativo de algoritmos según fases de mercado (aunque no es obligatorio). ([athenai.institute][1])

---

## 🧪 **Técnicas Permitidas y Recomendadas**

### 🛠️ **Libertad de enfoques**

* Puedes usar **cualquier técnica aprendida** en campos como: ([athenai.institute][1])

  * Finanzas cuantitativas tradicionales (regresión, técnicas econométricas). ([athenai.institute][1])
  * Machine learning clásico o avanzado (árboles, SVM, boosting). ([athenai.institute][1])
  * Algoritmos evolutivos o genéticos. ([athenai.institute][1])
  * Métodos basados en enjambres (swarm-based). ([athenai.institute][1])
  * Modelos híbridos que integren IA cuántica u otras técnicas avanzadas. ([athenai.institute][1])

### 🤖 **Recomendación explícita**

* Aunque no es obligatorio, **se recomienda considerar el uso de enfoques avanzados de aprendizaje por refuerzo** (Reinforcement Learning) para ciertos tipos de estrategia de inversión adaptativa. ([athenai.institute][1])

---

## 🚫 **Restricciones Importantes**

### 🔒 **Sin servicios externos en evaluación**

* Durante el entrenamiento de tu algoritmo, **puedes usar cualquier técnica o recurso legal que consideres útil**, pero: ([athenai.institute][1])

  * **En la fase de ejecución / evaluación final no se permite ningún elemento externo.** ([athenai.institute][1])
  * **Queda terminantemente prohibido realizar llamadas a APIs externas o servicios en línea** (incluidas las de IA generativa que requieran API KEY o acceso a modelos externos). ([athenai.institute][1])
  * Esto implica que durante la evaluación final, el modelo debe funcionar **autónomamente**, sin acceso a datos o servicios de terceros. ([athenai.institute][1])
  * Aunque el uso de IA generativa no está prohibido durante el diseño/training, **no puede formar parte del algoritmo en ejecución**. ([athenai.institute][1])

---

## 📈 **Evaluación del Modelo**

* La evaluación se realiza **después de que termine el plazo de la competición** (tres semanas desde la inscripción). ([athenai.institute][1])
* Un **comité de expertos** (incluyendo profesionales de la industria y académicos) evaluará las propuestas de los participantes. ([athenai.institute][1])
* **Criterios de evaluación** principales: ([athenai.institute][1])

  * **Coherencia** del algoritmo con las reglas inferidas del benchmark. ([athenai.institute][1])
  * **Rigor** técnico en construcción y justificación del modelo. ([athenai.institute][1])
  * **Adecuación** del algoritmo al escenario planteado y a las restricciones. ([athenai.institute][1])
* La ejecución del algoritmo se hace con **todos los datos (entrenamiento + evaluación)** completos. ([athenai.institute][1])

---

## 🏆 **Premios Técnicos (Condiciones)**

* Si tu modelo **supera al benchmark** y quedas entre los **60 mejores**, obtendrás una **beca de 6.875 €** distribuida así: ([athenai.institute][1])

  * **2.750 € para el programa “Quant Essential”**. ([athenai.institute][1])
  * **4.125 € para el programa “Top Quant”**. ([athenai.institute][1])
* **Validez de la beca:** solo para ediciones de **abril u octubre de 2026**. ([athenai.institute][1])
* **Requisito adicional:** para acceder al programa *Top Quant* en 2027, primero debes **haber completado y superado Quant Essential**. ([athenai.institute][1])

