import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, objective_function, bounds, population_size, generations,
                 crossover_rate, mutation_rate_gene, tournament_size,
                 blx_alpha=0.5, mutation_strength_factor=0.1, elitism_count=1):
        """
        Inicializa el Algoritmo Genético.

        Parámetros:
        - objective_function: La función a minimizar.
        - bounds: Un array de NumPy de forma (dim, 2) con los límites [min, max] para cada dimensión.
        - population_size: Número de individuos en la población.
        - generations: Número de generaciones a ejecutar.
        - crossover_rate: Probabilidad de cruce (pc).
        - mutation_rate_gene: Probabilidad de mutación por gen (pm).
        - tournament_size: Número de individuos en cada torneo de selección.
        - blx_alpha: Parámetro alfa para el cruce BLX-alpha.
        - mutation_strength_factor: Factor para determinar la desviación estándar de la mutación Gaussiana,
                                   relativo al rango de la variable.
        - elitism_count: Número de mejores individuos que pasan directamente a la siguiente generación.
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate_gene = mutation_rate_gene # Probabilidad de mutar cada gen individualmente
        self.tournament_size = tournament_size
        self.blx_alpha = blx_alpha
        self.mutation_strength_factor = mutation_strength_factor
        self.elitism_count = elitism_count

        if self.population_size % 2 != 0:
            # Asegurar que el tamaño de la población sea par para facilitar el cruce por pares
            self.population_size +=1
            print(f"Advertencia: El tamaño de la población se ajustó a {self.population_size} para que sea par.")


    def _initialize_population(self):
        """Inicializa la población con individuos aleatorios dentro de los límites."""
        population = []
        for _ in range(self.population_size):
            individual = np.zeros(self.dim)
            for i in range(self.dim):
                individual[i] = random.uniform(self.bounds[i, 0], self.bounds[i, 1])
            population.append(individual)
        return population

    def _evaluate_fitness(self, population):
        """Evalúa el fitness de cada individuo en la población."""
        fitness_values = []
        for individual in population:
            try:
                # Asegurarse de que el individuo esté dentro de los límites antes de evaluar
                # Esto es más un clamp, la mutación y cruce deberían intentar respetar los límites
                clamped_individual = np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])
                fitness_values.append(self.objective_function(clamped_individual))
            except Exception as e:
                print(f"Error evaluando individuo {individual}: {e}")
                fitness_values.append(float('inf')) # Penalización alta para individuos problemáticos
        return fitness_values

    def _tournament_selection(self, population, fitness_values):
        """Selecciona un padre usando selección por torneo."""
        selected_parents = []
        for _ in range(len(population)): # Seleccionar tantos padres como tamaño de población para crear la nueva generación
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            
            # El individuo con el mejor fitness (menor valor) gana el torneo
            winner_index_in_tournament = np.argmin(tournament_fitness)
            winner_index_in_population = tournament_indices[winner_index_in_tournament]
            selected_parents.append(population[winner_index_in_population])
        return selected_parents

    def _crossover_blx_alpha(self, parent1, parent2):
        """Realiza el cruce BLX-alpha entre dos padres."""
        child1 = np.zeros(self.dim)
        child2 = np.zeros(self.dim)
        for i in range(self.dim):
            d = abs(parent1[i] - parent2[i])
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            
            u1 = random.uniform(min_val - self.blx_alpha * d, max_val + self.blx_alpha * d)
            u2 = random.uniform(min_val - self.blx_alpha * d, max_val + self.blx_alpha * d)
            
            # Asegurar que los hijos estén dentro de los límites
            child1[i] = np.clip(u1, self.bounds[i, 0], self.bounds[i, 1])
            child2[i] = np.clip(u2, self.bounds[i, 0], self.bounds[i, 1])
        return child1, child2

    def _gaussian_mutation(self, individual):
        """Aplica mutación Gaussiana a un individuo."""
        mutated_individual = np.copy(individual)
        for i in range(self.dim):
            if random.random() < self.mutation_rate_gene:
                # La fuerza de la mutación (sigma) es un porcentaje del rango de la variable
                mutation_range = self.bounds[i, 1] - self.bounds[i, 0]
                sigma = self.mutation_strength_factor * mutation_range
                
                # Añadir ruido gaussiano
                gauss_val = random.gauss(0, sigma)
                mutated_individual[i] += gauss_val
                
                # Asegurar que el gen mutado esté dentro de los límites
                mutated_individual[i] = np.clip(mutated_individual[i], self.bounds[i, 0], self.bounds[i, 1])
        return mutated_individual

    def run(self):
        """Ejecuta el algoritmo genético."""
        population = self._initialize_population()
        fitness_values = self._evaluate_fitness(population)
        
        best_fitness_overall = float('inf')
        best_individual_overall = None
        convergence_history = [] # Almacena el mejor fitness de cada generación

        for generation in range(self.generations):
            # Elitismo: Guardar los mejores individuos
            sorted_indices = np.argsort(fitness_values)
            elite_individuals = [population[i] for i in sorted_indices[:self.elitism_count]]
            
            # Selección
            parents = self._tournament_selection(population, fitness_values)
            
            # Cruce y Mutación
            next_population = []
            
            # Mantener a los élites
            next_population.extend(elite_individuals)

            # Generar el resto de la población
            num_offspring_needed = self.population_size - self.elitism_count
            
            idx = 0
            while len(next_population) < self.population_size:
                parent1 = parents[idx % len(parents)] # Usar módulo para ciclar si es necesario
                parent2 = parents[(idx + 1) % len(parents)]
                idx += 2

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_blx_alpha(parent1, parent2)
                else:
                    child1, child2 = np.copy(parent1), np.copy(parent2) # Clonar padres si no hay cruce
                
                child1 = self._gaussian_mutation(child1)
                child2 = self._gaussian_mutation(child2)
                
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            population = next_population
            fitness_values = self._evaluate_fitness(population)
            
            current_best_fitness_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_fitness_idx]
            current_best_individual = population[current_best_fitness_idx]

            if current_best_fitness < best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_individual_overall = current_best_individual
            
            convergence_history.append(best_fitness_overall)
            
            if (generation + 1) % 10 == 0 or generation == 0 : # Imprimir progreso cada 10 generaciones
                print(f"Generación {generation + 1}/{self.generations} - Mejor Fitness: {best_fitness_overall:.6e}")

        return best_individual_overall, best_fitness_overall, convergence_history


if __name__ == '__main__':
    # Ejemplo de uso del Algoritmo Genético con una función simple (Sphere)
    print("\nProbando el Algoritmo Genético con la función Esfera:")
    
    def sphere_function(x):
        return np.sum(x**2)

    dim_sphere = 5
    bounds_sphere = np.array([[-5.12, 5.12]] * dim_sphere)
    
    ga_test = GeneticAlgorithm(
        objective_function=sphere_function,
        bounds=bounds_sphere,
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate_gene=0.1, # 10% de probabilidad de mutar cada gen
        tournament_size=3,
        blx_alpha=0.5,
        mutation_strength_factor=0.1, # Sigma = 10% del rango de la variable
        elitism_count=1
    )
    
    best_ind, best_fit, history = ga_test.run()
    
    print("\n--- Resultados del AG para la función Esfera ---")
    print(f"Mejor solución encontrada: {best_ind}")
    print(f"Mejor fitness encontrado: {best_fit:.6e}")

    # Para graficar la convergencia (requiere matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history)
        plt.title("Convergencia del AG para la función Esfera")
        plt.xlabel("Generación")
        plt.ylabel("Mejor Fitness")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib no está instalado. No se puede mostrar el gráfico de convergencia.")
        print("Datos de convergencia (primeros/últimos 10):")
        if len(history) > 20:
            print(history[:10])
            print("...")
            print(history[-10:])
        else:
            print(history)
