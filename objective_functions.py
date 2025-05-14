import numpy as np

# -----------------------------------------------------------------------------
# Definición de las Funciones Objetivo
# -----------------------------------------------------------------------------

def f1(x):
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

def f2(x):
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
        sum_term += (x[i]**2) * (2**(i+1)) # i va de 0 a 5, así que 2^(i+1) para potencias de 1 a 6
    return (1/899) * (sum_term - 1745)

def f3(x):
    """
    Función f3(x) = (x1^6 + x2^4 - 17)^2 + (2*x1 + x2 - 4)^2
    Dominio: -500 <= xi <= 500
    Dimensiones: 2 (x1, x2)
    Mínimo conocido: 0 en (x1, x2) = (1, 2) y otros puntos.
    """
    if len(x) != 2:
        raise ValueError("La función f3 requiere un vector de 2 dimensiones.")
    x1, x2 = x[0], x[1]
    term1 = x1**6 + x2**4 - 17
    term2 = 2*x1 + x2 - 4
    return term1**2 + term2**2

def f4(x):
    """
    Función f4(x) = sum_{i=1 to 10} [(ln(xi-2))^2 + (ln(10-xi))^2] - (prod_{i=1 to 10} xi)^0.2
    Dominio: 2.001 <= xi <= 9.999999 (ajustado para evitar ln(0) o ln(negativo))
    Dimensiones: 10
    Mínimo esperado: Cerca de x_i = 6 para todo i. Por ejemplo, f4(6,..,6) ~ 2.42
    """
    if len(x) != 10:
        raise ValueError("La función f4 requiere un vector de 10 dimensiones.")
    
    sum_logs_sq = 0
    prod_x = 1.0

    for val in x:
        if not (2.001 <= val <= 9.999999): # Verificación adicional de límites internos
             # Aplicar una penalización grande si está fuera de los límites seguros para logaritmos
            return 1e18 # Penalización muy alta
        
        term_ln_xi_minus_2 = np.log(val - 2)
        term_ln_10_minus_xi = np.log(10 - val)
        
        sum_logs_sq += term_ln_xi_minus_2**2 + term_ln_10_minus_xi**2
        prod_x *= val
        
    # Manejo de prod_x negativo si alguna vez ocurre (aunque los límites deberían prevenirlo)
    if prod_x < 0 and 0.2 % 1 != 0: # Raíz par de número negativo
        # Esto no debería ocurrir con los límites xi > 2
        # Si ocurre, es un problema numérico o una violación de límites no capturada
        # Se podría retornar una penalización muy alta o manejar el valor absoluto
        # Por ahora, asumimos que prod_x será positivo debido a los límites de x_i.
        # Si prod_x es muy cercano a cero, (prod_x)**0.2 puede ser problemático.
        # Sin embargo, x_i >= 2.001, por lo que prod_x será >= 2.001^10, que es grande.
        pass

    return sum_logs_sq - (prod_x**0.2)


# -----------------------------------------------------------------------------
# Información de las Funciones para el Optimizador
# -----------------------------------------------------------------------------

# Lista de diccionarios, cada uno describiendo una función objetivo
# Se utiliza para acceder fácilmente a las propiedades de cada función en el script principal.
OPTIMIZATION_PROBLEMS = [
    {
        "name": "f1",
        "function": f1,
        "dim": 2,
        "bounds": np.array([[-5.0, 5.0]] * 2) # Repite [-5, 5] para cada dimensión
    },
    {
        "name": "f2",
        "function": f2,
        "dim": 6,
        "bounds": np.array([[0.0, 1.0]] * 6)
    },
    {
        "name": "f3",
        "function": f3,
        "dim": 2,
        "bounds": np.array([[-500.0, 500.0]] * 2)
    },
    {
        "name": "f4",
        "function": f4,
        "dim": 10,
        # Límites ajustados para f4 para asegurar que los argumentos de logaritmo sean > 0
        # x_i - 2 > 0  => x_i > 2
        # 10 - x_i > 0 => x_i < 10
        "bounds": np.array([[2.001, 9.999999]] * 10)
    }
]

if __name__ == '__main__':
    # Código de prueba para las funciones (opcional)
    print("Probando las funciones objetivo:")

    # Prueba f1
    x_f1 = np.array([5.0, 0.0])
    print(f"f1({x_f1}) = {f1(x_f1)}") # Esperado: -516

    # Prueba f2
    x_f2 = np.array([0.0] * 6)
    print(f"f2({x_f2}) = {f2(x_f2)}") # Esperado: -1745/899 approx -1.94099

    # Prueba f3
    x_f3 = np.array([1.0, 2.0])
    print(f"f3({x_f3}) = {f3(x_f3)}") # Esperado: 0

    # Prueba f4
    x_f4 = np.array([6.0] * 10) # Todos los xi = 6
    print(f"f4({x_f4}) = {f4(x_f4)}") # Esperado: Aprox 2.42
    
    x_f4_lim_inf = np.array([2.001] * 10)
    print(f"f4({x_f4_lim_inf}) = {f4(x_f4_lim_inf)}")
    
    x_f4_lim_sup = np.array([9.999999] * 10)
    print(f"f4({x_f4_lim_sup}) = {f4(x_f4_lim_sup)}")

