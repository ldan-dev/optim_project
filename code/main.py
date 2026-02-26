import numpy as np
from wolfe_strong import wolfe_strong_cond

# Caso de prueba: f(x) = x^2 + y^2
class FuncionPrueba:
    def eval(self, x):
        return np.sum(x**2)

    def diff(self, x):
        return 2 * x

def ejecutar_optimizacion(func_objeto, x_inicial):
    # Punto actual xk
    xk = x_inicial
    
    # Direccion pk (negativo del gradiente)
    pk = -func_objeto.diff(xk) 
    
    # Tamaño de paso alpha
    alpha = 0.01
    
    # Llamada a la funcion
    cumple = wolfe_strong_cond(func_objeto, xk, alpha, pk)
    
    print(f"--- Resultados del Test ---")
    print(f"Punto xk: {xk}")
    print(f"Direccion pk: {pk}")
    print(f"Alpha: {alpha}")
    
    if cumple:
        print("Estado: ACEPTABLE (Cumple Strong Wolfe)")
    else:
        print("Estado: NO ACEPTABLE")

if __name__ == "__main__":
    f_instancia = FuncionPrueba()
    
    # Ejemplo con un punto xk especifico
    x0 = np.array([1.0, 5.0])
    
    ejecutar_optimizacion(f_instancia, x0)