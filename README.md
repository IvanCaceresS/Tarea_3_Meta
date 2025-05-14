# Tarea_3_Meta

Este proyecto implementa un Algoritmo Genético (AG) para encontrar los mínimos de funciones multimodales y compara sus resultados con los obtenidos mediante optimizadores matemáticos.

## Descripción de Archivos

* **`objective_functions.py`**: Define las cuatro funciones objetivo (f1, f2, f3, f4) propuestas en la tarea, junto con sus dimensiones y límites.
* **`genetic_algorithm.py`**: Contiene la implementación de la clase `GeneticAlgorithm`. Incluye la inicialización de la población, evaluación del fitness, selección por torneo, cruce BLX-alpha, mutación Gaussiana y elitismo.
* **`main_experiment.py`**: Orquesta la ejecución de los experimentos. Para cada función objetivo y cada configuración de parámetros del AG:
    * Realiza 10 ejecuciones independientes del AG.
    * Registra el mejor fitness, el mejor individuo y el tiempo de ejecución para cada corrida en un archivo CSV (`Resultados_AG/Resultados_CSV/resultados_experimentos.csv`).
    * Calcula y muestra estadísticas (mínimo, máximo, promedio, desviación estándar del fitness) para el conjunto de 10 ejecuciones.
    * Genera y guarda gráficos de convergencia promedio (si `matplotlib` está disponible) en la carpeta `Resultados_AG/Graficos_Convergencia/`.
    * Muestra resúmenes del mejor fitness encontrado para cada función y un resumen global al final.
* **`minimos_teoricos.py`**: Calcula los mínimos de las funciones objetivo utilizando el método `minimize` de la biblioteca `scipy.optimize` como punto de referencia y validación.
* **`Resultados_AG/`**: Carpeta generada por `main_experiment.py` que contiene:
    * `Resultados_CSV/resultados_experimentos.csv`: Archivo CSV con los datos detallados de cada ejecución.
    * `Graficos_Convergencia/`: Carpeta donde se guardan los gráficos de convergencia.

## Configuraciones del Algoritmo Genético y Justificación de Parámetros

Se definieron 4 configuraciones de parámetros distintas para el Algoritmo Genético, con el objetivo de analizar cómo diferentes estrategias y valores afectan el rendimiento en la optimización de las funciones multimodales. Todas las configuraciones utilizan selección por torneo, cruce BLX-alpha y mutación Gaussiana.

A continuación, se detalla cada configuración y la justificación de los valores de sus parámetros:

---

### Configuración 1: "Config_1_Baseline"

* **Justificación General:** Esta configuración sirve como un punto de partida estándar, utilizando valores de parámetros que son comúnmente recomendados o que han demostrado ser efectivos en una variedad de problemas de optimización con AGs.
* **Parámetros:**
    * `population_size: 50`: Un tamaño de población moderado. Suficientemente grande para mantener diversidad, pero no tanto como para ralentizar excesivamente las primeras pruebas.
    * `generations: 100`: Un número de generaciones que permite una convergencia razonable para muchos problemas sin un costo computacional excesivo para una configuración base.
    * `crossover_rate: 0.8`: Una alta probabilidad de cruce (80%) es común, fomentando la recombinación de buenas características de los padres.
    * `mutation_rate_gene: 0.1`: Probabilidad de mutación por gen del 10%. Busca introducir nueva información genética y evitar la convergencia prematura, sin ser tan alta como para perturbar excesivamente la búsqueda.
    * `tournament_size: 3`: Un tamaño de torneo pequeño (3 individuos) implica una presión de selección moderada. Da a individuos ligeramente peores una oportunidad de ser seleccionados, manteniendo la diversidad.
    * `blx_alpha: 0.5`: Un valor estándar para BLX-alpha que permite a los hijos generarse en un rango que se extiende más allá del hiperrectángulo definido por los padres, promoviendo la exploración.
    * `mutation_strength_factor: 0.1`: La desviación estándar de la mutación Gaussiana es el 10% del rango de la variable. Esto permite mutaciones que no son ni demasiado pequeñas (inefectivas) ni demasiado grandes (aleatorias).
    * `elitism_count: 1`: Se preserva el mejor individuo de la generación actual y pasa directamente a la siguiente. Asegura que el mejor resultado encontrado hasta el momento no se pierda.

---

### Configuración 2: "Config_2_LargerPop_MoreGens"

* **Justificación General:** Aumentar el tamaño de la población y el número de generaciones para permitir una exploración más exhaustiva del espacio de búsqueda y una explotación más refinada de las regiones prometedoras. Esto es especialmente útil para funciones complejas o de alta dimensionalidad.
* **Parámetros:**
    * `population_size: 100`: Se duplica el tamaño de la población respecto a la base. Una población más grande puede cubrir más regiones del espacio de búsqueda simultáneamente y es menos propensa a la convergencia prematura.
    * `generations: 150`: Se aumenta el número de generaciones para dar más tiempo al algoritmo para converger, especialmente con una población más grande.
    * `crossover_rate: 0.8`: Se mantiene igual que la base, ya que es un valor generalmente bueno.
    * `mutation_rate_gene: 0.1`: Se mantiene igual, asumiendo que la mayor población ya contribuye a la diversidad.
    * `tournament_size: 3`: Se mantiene la presión de selección moderada.
    * `blx_alpha: 0.5`: Se mantiene el valor estándar.
    * `mutation_strength_factor: 0.1`: Se mantiene la fuerza de mutación.
    * `elitism_count: 2`: Se aumenta ligeramente el elitismo para asegurar la preservación de los dos mejores individuos, lo cual es razonable con una población mayor.

---

### Configuración 3: "Config_3_HigherMutation"

* **Justificación General:** Incrementar la tasa y la fuerza de la mutación para aumentar la diversidad genética en la población. Esto puede ayudar al algoritmo a escapar de óptimos locales y explorar nuevas áreas del espacio de búsqueda, lo cual es crucial para funciones multimodales.
* **Parámetros:**
    * `population_size: 50`: Se mantiene el tamaño de población base para aislar el efecto del aumento de la mutación.
    * `generations: 100`: Se mantiene el número de generaciones base.
    * `crossover_rate: 0.8`: Se mantiene la tasa de cruce estándar.
    * `mutation_rate_gene: 0.25`: Se aumenta significativamente la probabilidad de mutación por gen (al 25%). Esto introduce variabilidad de forma más agresiva.
    * `tournament_size: 3`: Presión de selección moderada.
    * `blx_alpha: 0.5`: Valor estándar.
    * `mutation_strength_factor: 0.15`: Se aumenta ligeramente la fuerza de la mutación (desviación estándar al 15% del rango de la variable), permitiendo saltos un poco más grandes durante la mutación.
    * `elitism_count: 1`: Elitismo estándar.

---

### Configuración 4: "Config_4_DifferentStrategy"

* **Justificación General:** Probar una estrategia alternativa con una mayor presión de selección, un cruce más frecuente pero más conservador (hijos más cercanos a los padres), y una mutación más fina y menos frecuente. El objetivo es una explotación más intensiva una vez que se identifican regiones prometedoras.
* **Parámetros:**
    * `population_size: 70`: Un tamaño de población intermedio.
    * `generations: 120`: Un número de generaciones intermedio.
    * `crossover_rate: 0.9`: Tasa de cruce muy alta (90%) para maximizar la recombinación.
    * `mutation_rate_gene: 0.05`: Tasa de mutación por gen más baja (5%). Se confía más en el cruce para la exploración inicial y en la selección para la explotación.
    * `tournament_size: 5`: Se aumenta el tamaño del torneo a 5, lo que incrementa la presión de selección (es más probable que los mejores individuos sean seleccionados).
    * `blx_alpha: 0.3`: Un valor de alfa más pequeño para BLX-alpha (0.3). Esto tiende a generar hijos más cercanos al hiperrectángulo definido por los padres, lo que significa un cruce más conservador y enfocado en la explotación local.
    * `mutation_strength_factor: 0.05`: Mutación más fina (desviación estándar al 5% del rango), para realizar ajustes pequeños alrededor de las soluciones prometedoras.
    * `elitism_count: 1`: Elitismo estándar.

---
