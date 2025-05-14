import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Definición de las Funciones Objetivo (igual que en objective_functions.py)
# -----------------------------------------------------------------------------

def f1_opt(x):
    """
    Función f1(x) = 4 - 4*x1^3 - 4*x1 + x2^2
    Dominio: -5 <= xi <= 5
    Dimensiones: 2 (x1, x2)
    Mínimo conocido: -516 en (x1, x2) = (5, 0)
    """
    if len(x) != 2:
        raise ValueError("La función f1 requiere un vector de 2 dimensiones.")
    x1, x2 = x[0], x[1]
    return 4 - 4*x1**3 - 4*x1 + x2**2

def f2_opt(x):
    """
    Función f2(x) = (1/899) * (sum_{i=1 to 6} (x_i^2 * 2^i) - 1745)
    Dominio: 0 <= xi <= 1
    Dimensiones: 6
    Mínimo conocido: -1745/899 approx -1.94099 en x_i = 0 para todo i.
    """
    if len(x) != 6:
        raise ValueError("La función f2 requiere un vector de 6 dimensiones.")
    sum_term = 0
    for i in range(6):
        sum_term += (x[i]**2) * (2**(i+1)) 
    return (1/899) * (sum_term - 1745)

def f3_opt(x):
    """
    Función f3(x) = (x1^6 + x2^4 - 17)^2 + (2*x1 + x2 - 4)^2
    Dominio: -500 <= xi <= 500
    Dimensiones: 2 (x1, x2)
    Mínimo conocido: 0 en (x1, x2) = (1, 2) y otros puntos como (1.5968, 0.8064).
    """
    if len(x) != 2:
        raise ValueError("La función f3 requiere un vector de 2 dimensiones.")
    x1, x2 = x[0], x[1]
    term1 = x1**6 + x2**4 - 17
    term2 = 2*x1 + x2 - 4
    return term1**2 + term2**2

def f4_opt(x):
    """
    Función f4(x) = sum_{i=1 to 10} [(ln(xi-2))^2 + (ln(10-xi))^2] - (prod_{i=1 to 10} xi)^0.2
    Dominio: 2.001 <= xi <= 9.999999
    Dimensiones: 10
    """
    if len(x) != 10:
        raise ValueError("La función f4 requiere un vector de 10 dimensiones.")
    
    # Asegurar que los valores estén dentro de los límites para evitar errores de logaritmo
    for val_check in x:
        if not (2.0 < val_check < 10.0):
            # Retornar un valor muy alto si está fuera de los límites estrictos para el logaritmo
            # Esto ayuda al optimizador a evitar regiones problemáticas.
            return 1e18 

    sum_logs_sq = 0
    prod_x = 1.0

    for val in x:
        # Pequeño ajuste para evitar problemas en los bordes exactos con logaritmos
        # Aunque los bounds deberían manejar esto, una doble verificación no hace daño.
        safe_val_minus_2 = max(val - 2, 1e-9) # Evita log(0) o log(negativo)
        safe_10_minus_val = max(10 - val, 1e-9) # Evita log(0) o log(negativo)

        term_ln_xi_minus_2 = np.log(safe_val_minus_2)
        term_ln_10_minus_xi = np.log(safe_10_minus_val)
        
        sum_logs_sq += term_ln_xi_minus_2**2 + term_ln_10_minus_xi**2
        prod_x *= val
        
    if prod_x < 0 and 0.2 % 1 != 0: # Manejo de raíz par de número negativo
        # Esto no debería ocurrir con los límites xi > 2
        return 1e18 # Penalización alta
    
    # Si prod_x es muy pequeño y positivo, (prod_x)**0.2 es válido.
    # Si prod_x es cero, el resultado es cero.
    # Si prod_x es negativo, y la potencia es fraccional, puede dar complejo.
    # Con los límites xi > 2, prod_x siempre será positivo.
    
    return sum_logs_sq - (np.abs(prod_x)**0.2 if prod_x >=0 else -np.abs(prod_x)**0.2)


# -----------------------------------------------------------------------------
# Definición de Límites y Puntos de Inicio
# -----------------------------------------------------------------------------

problems_to_solve = [
    {
        "name": "f1",
        "function": f1_opt,
        "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
        "x0": [0.0, 0.0] # Punto de inicio
    },
    {
        "name": "f2",
        "function": f2_opt,
        "bounds": [(0.0, 1.0)] * 6,
        "x0": [0.5] * 6 # Punto de inicio en el centro del dominio
    },
    {
        "name": "f3_opt1", # Probando un óptimo conocido
        "function": f3_opt,
        "bounds": [(-500.0, 500.0), (-500.0, 500.0)],
        "x0": [1.0, 2.0] 
    },
    {
        "name": "f3_opt2", # Probando otro óptimo conocido
        "function": f3_opt,
        "bounds": [(-500.0, 500.0), (-500.0, 500.0)],
        "x0": [1.5968, 0.8064] 
    },
     {
        "name": "f3_center", # Probando desde el centro
        "function": f3_opt,
        "bounds": [(-500.0, 500.0), (-500.0, 500.0)],
        "x0": [0.0, 0.0] 
    },
    {
        "name": "f4",
        "function": f4_opt,
        "bounds": [(2.001, 9.999999)] * 10,
        # Usamos un punto de inicio basado en los resultados del AG, que fue muy bueno.
        # O el centro del dominio: (2.001 + 9.999999) / 2 = 6.0004995
        "x0": [9.35] * 10 
        # "x0": [6.0] * 10 # Alternativa: centro del dominio
    }
]

# -----------------------------------------------------------------------------
# Ejecución de la Optimización
# -----------------------------------------------------------------------------

print("Calculando mínimos teóricos con scipy.optimize.minimize:\n")

for problem in problems_to_solve:
    print(f"--- Función: {problem['name']} ---")
    
    # Métodos comunes que soportan límites: 'L-BFGS-B', 'TNC', 'SLSQP'
    # 'L-BFGS-B' es una buena opción para optimización no lineal con límites.
    result = minimize(
        problem["function"],
        problem["x0"],
        method='L-BFGS-B', # También puedes probar 'SLSQP' o 'TNC'
        bounds=problem["bounds"],
        options={'disp': False, 'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8} # Opciones para mayor precisión y más iteraciones
    )
    
    if result.success:
        print(f"  Éxito: {result.success}")
        print(f"  Mínimo encontrado en x = {np.array2string(result.x, formatter={'float_kind':lambda x: '%.4f' % x})}")
        print(f"  Valor mínimo de la función f(x) = {result.fun:.6e}")
        # print(f"  Mensaje: {result.message}")
        # print(f"  Número de evaluaciones de la función: {result.nfev}")
        # print(f"  Número de iteraciones: {result.nit}")
    else:
        print(f"  La optimización NO tuvo éxito para {problem['name']}.")
        print(f"  Mensaje: {result.message}")
        print(f"  Valor de la función en el último punto: {result.fun:.6e}")
        print(f"  Último punto x: {np.array2string(result.x, formatter={'float_kind':lambda x: '%.4f' % x})}")
    print("-" * 30)

