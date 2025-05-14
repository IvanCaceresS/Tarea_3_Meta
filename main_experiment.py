import numpy as np
import time
import csv # Importado para la funcionalidad CSV
import os  # Importado para crear directorios

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
GA_CONFIGURATIONS = [
    {
        "name": "Config_1_Baseline",
        "params": {
            "population_size": 50,
            "generations": 100,
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.1,
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.1,
            "elitism_count": 1
        },
        "justification": "Configuración base con parámetros comúnmente utilizados."
    },
    {
        "name": "Config_2_LargerPop_MoreGens",
        "params": {
            "population_size": 100,
            "generations": 150,
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.1,
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.1,
            "elitism_count": 2
        },
        "justification": "Mayor población y generaciones para más exploración y explotación."
    },
    {
        "name": "Config_3_HigherMutation",
        "params": {
            "population_size": 50,
            "generations": 100,
            "crossover_rate": 0.8,
            "mutation_rate_gene": 0.25,
            "tournament_size": 3,
            "blx_alpha": 0.5,
            "mutation_strength_factor": 0.15,
            "elitism_count": 1
        },
        "justification": "Mayor tasa de mutación para incrementar la diversidad."
    },
    {
        "name": "Config_4_DifferentStrategy",
        "params": {
            "population_size": 70,
            "generations": 120,
            "crossover_rate": 0.9,
            "mutation_rate_gene": 0.05,
            "tournament_size": 5,
            "blx_alpha": 0.3,
            "mutation_strength_factor": 0.05,
            "elitism_count": 1
        },
        "justification": "Mayor presión de selección, cruce más frecuente pero conservador."
    }
]

# Número de ejecuciones para cada función y configuración
NUM_EXECUTIONS = 10

# Rutas para guardar resultados
RESULTS_DIR = "Resultados_AG" 
CSV_DIR = os.path.join(RESULTS_DIR, "Resultados_CSV")
PLOTS_DIR = os.path.join(RESULTS_DIR, "Graficos_Convergencia")
CSV_FILENAME = os.path.join(CSV_DIR, "resultados_experimentos.csv")

# -----------------------------------------------------------------------------
# Script Principal para Ejecutar los Experimentos
# -----------------------------------------------------------------------------

def run_experiments():
    """
    Ejecuta los experimentos de optimización para cada función y configuración del AG.
    Guarda los resultados en un archivo CSV, los gráficos de convergencia,
    e imprime resúmenes del mejor fitness encontrado.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    start_total_time = time.time()

    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Funcion", "Configuracion", "Ejecucion_Num", 
            "Mejor_Fitness", "Tiempo_Ejecucion_s", "Mejor_Individuo"
        ])

        grand_overall_best_results = {} 

        for problem_info in OPTIMIZATION_PROBLEMS:
            func_name = problem_info["name"]
            objective_func = problem_info["function"]
            dim = problem_info["dim"]
            bounds = problem_info["bounds"]

            print(f"\n{'='*80}")
            print(f"Optimizando Función: {func_name.upper()} (Dimensiones: {dim})")
            print(f"{'='*80}")

            best_fitness_for_current_function = float('inf')
            best_individual_for_current_function = None
            best_config_for_current_function = ""

            for config_info in GA_CONFIGURATIONS:
                config_name = config_info["name"]
                ga_params = config_info["params"]
                justification = config_info["justification"]

                print(f"\n--- Configuración del AG: {config_name} ---")
                print(f"    Justificación: {justification}")

                all_runs_best_fitness = []
                all_runs_best_individuals = []
                all_runs_convergence_histories = []
                
                print(f"    Realizando {NUM_EXECUTIONS} ejecuciones...")
                start_config_total_time = time.time()

                for i in range(NUM_EXECUTIONS):
                    print(f"\n      Ejecución {i + 1}/{NUM_EXECUTIONS} para {func_name} con {config_name}")
                    
                    start_execution_time = time.time()
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
                    end_execution_time = time.time()
                    execution_time_s = end_execution_time - start_execution_time

                    all_runs_best_fitness.append(best_fitness)
                    all_runs_best_individuals.append(best_individual)
                    all_runs_convergence_histories.append(convergence_history)
                    
                    csv_writer.writerow([
                        func_name, config_name, i + 1, 
                        f"{best_fitness:.6e}", 
                        f"{execution_time_s:.2f}", 
                        np.array2string(best_individual, formatter={'float_kind':lambda x: "%.4f" % x})
                    ])
                    print(f"      Fin Ejecución {i + 1}: Mejor Fitness = {best_fitness:.6e}, Tiempo: {execution_time_s:.2f}s")

                end_config_total_time = time.time()
                total_time_for_config = end_config_total_time - start_config_total_time
                print(f"\n    Tiempo total para {NUM_EXECUTIONS} ejecuciones con {config_name} en {func_name}: {total_time_for_config:.2f}s")

                min_fitness_this_config = np.min(all_runs_best_fitness)
                max_fitness_this_config = np.max(all_runs_best_fitness)
                mean_fitness_this_config = np.mean(all_runs_best_fitness)
                std_fitness_this_config = np.std(all_runs_best_fitness)
                idx_best_this_config = np.argmin(all_runs_best_fitness)
                best_individual_this_config = all_runs_best_individuals[idx_best_this_config]

                print("\n    --- Resultados Estadísticos para esta Configuración ({config_name}) ---")
                print(f"    Mejor Fitness (Mínimo de {NUM_EXECUTIONS} ejecuciones): {min_fitness_this_config:.6e}")
                print(f"    Mejor Individuo encontrado en estas {NUM_EXECUTIONS} ejecuciones:")
                # Formatear individuo para impresión en consola, acortando si es necesario
                if len(best_individual_this_config) > 6:
                     print(f"        [{' '.join(f'{x:.4f}' for x in best_individual_this_config[:3])} ... {' '.join(f'{x:.4f}' for x in best_individual_this_config[-3:])}]")
                else:
                     print(f"        [{' '.join(f'{x:.4f}' for x in best_individual_this_config)}]")
                print(f"    Peor Fitness (Máximo de {NUM_EXECUTIONS} ejecuciones): {max_fitness_this_config:.6e}")
                print(f"    Fitness Promedio: {mean_fitness_this_config:.6e}")
                print(f"    Desviación Estándar del Fitness: {std_fitness_this_config:.6e}")

                if min_fitness_this_config < best_fitness_for_current_function:
                    best_fitness_for_current_function = min_fitness_this_config
                    best_individual_for_current_function = best_individual_this_config
                    best_config_for_current_function = config_name
                
                if MATPLOTLIB_AVAILABLE and all_runs_convergence_histories:
                    try:
                        convergence_array = np.array(all_runs_convergence_histories)
                        avg_convergence = np.mean(convergence_array, axis=0)
                        std_convergence = np.std(convergence_array, axis=0)
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
                        plot_filename = os.path.join(PLOTS_DIR, f"convergencia_{func_name}_{config_name}.png")
                        plt.savefig(plot_filename)
                        print(f"    Gráfico de convergencia guardado como: {plot_filename}")
                        plt.close() 
                    except Exception as e:
                        print(f"    ERROR: No se pudo generar o guardar el gráfico para {func_name} con {config_name}.")
                        print(f"    Detalle del error: {e}")
                elif not MATPLOTLIB_AVAILABLE and all_runs_convergence_histories:
                    print(f"    ADVERTENCIA: Matplotlib no está disponible. Mostrando datos de convergencia en lugar de gráfico.")
                    # (código para imprimir datos de convergencia si no hay matplotlib)
                print("-" * 60)

            print(f"\n{'*'*70}")
            print(f"RESUMEN FINAL PARA LA FUNCIÓN: {func_name.upper()}")
            print(f"  El mejor fitness global encontrado para '{func_name}' en todas las configuraciones fue: {best_fitness_for_current_function:.6e}")
            print(f"  Este resultado fue obtenido con la configuración: '{best_config_for_current_function}'")
            print(f"  El mejor individuo global correspondiente para '{func_name}' es:")
            if best_individual_for_current_function is not None:
                if len(best_individual_for_current_function) > 6:
                    print(f"    [{' '.join(f'{x:.4f}' for x in best_individual_for_current_function[:3])} ... {' '.join(f'{x:.4f}' for x in best_individual_for_current_function[-3:])}]")
                else:
                    print(f"    [{' '.join(f'{x:.4f}' for x in best_individual_for_current_function)}]")
            else:
                print("    No se encontró un individuo.")
            print(f"{'*'*70}\n")

            grand_overall_best_results[func_name] = {
                'best_fitness': best_fitness_for_current_function,
                'best_individual': best_individual_for_current_function,
                'config_name': best_config_for_current_function
            }

    end_total_time = time.time()
    print(f"\n{'='*80}")
    print(f"Tiempo total de todos los experimentos: {end_total_time - start_total_time:.2f} segundos.")
    print(f"Archivo CSV con resultados detallados guardado en: {CSV_FILENAME}")
    print(f"Los gráficos (si se generaron) están en la carpeta: {PLOTS_DIR}")
    
    print(f"\n{'='*80}")
    print("RESUMEN GENERAL DE MEJORES RESULTADOS GLOBALES POR FUNCIÓN")
    print(f"(Considerando todas las configuraciones y ejecuciones)")
    print(f"{'='*80}")
    for func_key, results_val in grand_overall_best_results.items():
        print(f"Función: {func_key.upper()}")
        print(f"  Mejor Fitness Global Absoluto: {results_val['best_fitness']:.6e}")
        print(f"  Obtenido con la Configuración: '{results_val['config_name']}'")
        print(f"  Mejor Individuo Global Absoluto:")
        if results_val['best_individual'] is not None:
            if len(results_val['best_individual']) > 6:
                 print(f"    [{' '.join(f'{x:.4f}' for x in results_val['best_individual'][:3])} ... {' '.join(f'{x:.4f}' for x in results_val['best_individual'][-3:])}]")
            else:
                 print(f"    [{' '.join(f'{x:.4f}' for x in results_val['best_individual'])}]")
        else:
            print("    No se encontró individuo.")
        print("-" * 50)
    print(f"{'='*80}")

if __name__ == "__main__":
    run_experiments()
