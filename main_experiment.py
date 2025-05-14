import numpy as np
import time
from objective_functions import OPTIMIZATION_PROBLEMS # Importa las funciones y sus detalles
from genetic_algorithm import GeneticAlgorithm

# Intentar importar matplotlib para gráficos
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Advertencia: Matplotlib no está instalado. Los gráficos de convergencia no se generarán automáticamente.")
    print("Se imprimirán los datos de convergencia en su lugar.")

# -----------------------------------------------------------------------------
# Configuraciones de Parámetros para el Algoritmo Genético
# -----------------------------------------------------------------------------
# Se deben definir al menos 4 configuraciones distintas con justificación.

GA_CONFIGURATIONS = [
    {
        "name": "Config_1_Baseline",
        "params": {
            "population_size": 50,
            "generations": 100, # Ajustado para que no tarde demasiado en la demo
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.1, # Probabilidad de mutación por gen
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.1, # std dev = 10% del rango de la variable
            "elitism_count": 1
        },
        "justification": "Configuración base con parámetros comúnmente utilizados."
    },
    {
        "name": "Config_2_LargerPop_MoreGens",
        "params": {
            "population_size": 100, # Población más grande
            "generations": 150,    # Más generaciones
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.1,
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.1,
            "elitism_count": 2 # Un poco más de elitismo para población mayor
        },
        "justification": "Mayor población y generaciones para más exploración y explotación, buscando mejores soluciones."
    },
    {
        "name": "Config_3_HigherMutation",
        "params": {
            "population_size": 50,
            "generations": 100,
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.25, # Tasa de mutación por gen más alta
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.15, # Mutación un poco más fuerte
            "elitism_count": 1
        },
        "justification": "Mayor tasa de mutación para incrementar la diversidad y ayudar a escapar de óptimos locales."
    },
    {
        "name": "Config_4_DifferentStrategy",
        "params": {
            "population_size": 70, # Tamaño intermedio
            "generations": 120,
            "crossover_rate": 0.9,  # Cruce más frecuente
            "mutation_rate_gene": 0.05, # Mutación por gen más baja pero...
            "tournament_size": 5,   # Mayor presión de selección
            "blx_alpha": 0.3,       # Cruce BLX más restrictivo (hijos más cercanos a los padres)
            "mutation_strength_factor": 0.05, # Mutación más fina
            "elitism_count": 1
        },
        "justification": "Probar una estrategia con mayor presión de selección, cruce más frecuente pero más conservador, y mutación más fina."
    }
]

# Número de ejecuciones para cada función y configuración
NUM_EXECUTIONS = 10 # Como se solicita en la tarea

# -----------------------------------------------------------------------------
# Script Principal para Ejecutar los Experimentos
# -----------------------------------------------------------------------------

def run_experiments():
    """
    Ejecuta los experimentos de optimización para cada función y configuración del AG.
    """
    start_total_time = time.time()

    for problem_info in OPTIMIZATION_PROBLEMS:
        func_name = problem_info["name"]
        objective_func = problem_info["function"]
        dim = problem_info["dim"]
        bounds = problem_info["bounds"]

        print(f"\n{'='*80}")
        print(f"Optimizando Función: {func_name} (Dimensiones: {dim})")
        print(f"{'='*80}")

        for config_info in GA_CONFIGURATIONS:
            config_name = config_info["name"]
            ga_params = config_info["params"]
            justification = config_info["justification"]

            print(f"\n--- Configuración del AG: {config_name} ---")
            print(f"    Justificación: {justification}")
            #for p_name, p_val in ga_params.items(): print(f"    {p_name}: {p_val}")


            all_runs_best_fitness = []
            all_runs_best_individuals = []
            all_runs_convergence_histories = [] # Para graficar la convergencia promedio

            print(f"    Realizando {NUM_EXECUTIONS} ejecuciones...")
            start_config_time = time.time()

            for i in range(NUM_EXECUTIONS):
                print(f"\n    Ejecución {i + 1}/{NUM_EXECUTIONS} para {func_name} con {config_name}")
                
                # Crear una instancia del AG para cada ejecución (para resetear estado si es necesario)
                ga = GeneticAlgorithm(
                    objective_function=objective_func,
                    bounds=bounds,
                    population_size=ga_params["population_size"],
                    generations=ga_params["generations"],
                    crossover_rate=ga_params["crossover_rate"],
                    mutation_rate_gene=ga_params["mutation_rate_gene"],
                    tournament_size=ga_params["tournament_size"],
                    blx_alpha=ga_params["blx_alpha"],
                    mutation_strength_factor=ga_params["mutation_strength_factor"],
                    elitism_count=ga_params["elitism_count"]
                )
                
                best_individual, best_fitness, convergence_history = ga.run()
                
                all_runs_best_fitness.append(best_fitness)
                all_runs_best_individuals.append(best_individual)
                all_runs_convergence_histories.append(convergence_history)
                
                print(f"    Fin Ejecución {i + 1}: Mejor Fitness = {best_fitness:.6e}")
                # print(f"    Mejor Individuo: {best_individual}")


            end_config_time = time.time()
            print(f"\n    Tiempo para {NUM_EXECUTIONS} ejecuciones con {config_name} en {func_name}: {end_config_time - start_config_time:.2f}s")

            # Calcular estadísticas de las 10 ejecuciones
            min_fitness = np.min(all_runs_best_fitness)
            max_fitness = np.max(all_runs_best_fitness)
            mean_fitness = np.mean(all_runs_best_fitness)
            std_fitness = np.std(all_runs_best_fitness)
            
            best_overall_idx = np.argmin(all_runs_best_fitness)
            best_overall_individual_for_config = all_runs_best_individuals[best_overall_idx]


            print("\n    --- Resultados Estadísticos para esta Configuración ---")
            print(f"    Mejor Fitness (Mínimo de {NUM_EXECUTIONS} ejecuciones): {min_fitness:.6e}")
            print(f"    Peor Fitness (Máximo de {NUM_EXECUTIONS} ejecuciones): {max_fitness:.6e}")
            print(f"    Fitness Promedio: {mean_fitness:.6e}")
            print(f"    Desviación Estándar del Fitness: {std_fitness:.6e}")
            print(f"    Mejor individuo global para esta config (fitness {min_fitness:.6e}):")
            # Imprimir solo las primeras y últimas coordenadas si es muy largo
            if len(best_overall_individual_for_config) > 6:
                 print(f"        {best_overall_individual_for_config[:3]} ... {best_overall_individual_for_config[-3:]}")
            else:
                 print(f"        {best_overall_individual_for_config}")


            # Graficar la convergencia promedio para esta configuración y función
            if MATPLOTLIB_AVAILABLE and all_runs_convergence_histories:
                # Asegurar que todas las historias de convergencia tengan la misma longitud (num_generations)
                # Esto debería ser así por defecto si generations es fijo.
                # Si alguna ejecución terminara antes por algún criterio (no implementado aquí), se necesitaría padding.
                
                # Convertir lista de listas a array 2D para facilitar el promedio por columnas (generaciones)
                convergence_array = np.array(all_runs_convergence_histories)
                avg_convergence = np.mean(convergence_array, axis=0)
                std_convergence = np.std(convergence_array, axis=0) # Opcional: para bandas de error

                plt.figure(figsize=(10, 6))
                plt.plot(avg_convergence, label="Fitness Promedio")
                plt.fill_between(range(len(avg_convergence)), 
                                 avg_convergence - std_convergence, 
                                 avg_convergence + std_convergence, 
                                 alpha=0.2, label="Std Dev")
                plt.title(f"Convergencia Promedio del AG\nFunción: {func_name} - Config: {config_name}")
                plt.xlabel("Generación")
                plt.ylabel("Mejor Fitness Promedio")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # Guardar el gráfico en un archivo
                plot_filename = f"convergencia_{func_name}_{config_name}.png"
                plt.savefig(plot_filename)
                print(f"    Gráfico de convergencia guardado como: {plot_filename}")
                # plt.show() # Descomentar si se desea mostrar interactivamente (puede pausar el script)
                plt.close()

            elif not MATPLOTLIB_AVAILABLE and all_runs_convergence_histories:
                print(f"    Datos de convergencia promedio (primeros 5 y últimos 5 valores de {ga_params['generations']} generaciones):")
                try:
                    convergence_array = np.array(all_runs_convergence_histories)
                    avg_convergence = np.mean(convergence_array, axis=0)
                    if len(avg_convergence) > 10:
                        print(f"        {avg_convergence[:5]}")
                        print("        ...")
                        print(f"        {avg_convergence[-5:]}")
                    else:
                        print(f"        {avg_convergence}")
                except Exception as e:
                    print(f"        No se pudieron procesar los datos de convergencia: {e}")
            print("-" * 60)

    end_total_time = time.time()
    print(f"\n{'='*80}")
    print(f"Tiempo total de todos los experimentos: {end_total_time - start_total_time:.2f} segundos.")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_experiments()

