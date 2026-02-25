"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 25/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
Implementacion de la funcion de Griewangk

Funcion de Griewangk:
    f(x) = Σ(x_i²/4000) - Π(cos(x_i/√i)) + 1
    
    donde i = 1, 2, ..., n
    
Minimo global: f(0, 0, ..., 0) = 0
Dominio tipico: [-600, 600]^n
"""

import numpy as np
from function import Function
from plot import Plot


class Func_Griew(Function):
    """
    Griewangk function
    
    Caracteristicas:
    - Multimodal con muchos minimos locales
    - Minimo global en x = (0, 0, ..., 0) con f(x*) = 0
    - Útil para probar algoritmos de optimizacion global
    """
    
    def __init__(self, params=None):
        """constructor"""
        super().__init__(params)
        self.name = "Griewangk"
        self.limits = [-600, 600]  # dominio tipico
        self.path = []

    def eval(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewangk function at point x
        
        f(x) = Σ(x_i²/4000) - Π(cos(x_i/√i)) + 1
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        # Termino de suma
        sum_term = np.sum(x**2) / 4000
        
        # Termino de producto: Π(cos(x_i/√i))
        # i va de 1 a n (indice basado en 1)
        i = np.arange(1, n + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        
        return sum_term - prod_term + 1

    def diff(self, x: np.ndarray) -> np.ndarray:
        """
        Return the gradient vector at x (1st derivative)
        
        ∂f/∂x_k = x_k/2000 + (1/√k) * tan(x_k/√k) * Π_{i=1}^{n}(cos(x_i/√i))
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        i = np.arange(1, n + 1)
        sqrt_i = np.sqrt(i)
        
        # Producto completo de todos los cosenos
        prod_cos = np.prod(np.cos(x / sqrt_i))
        
        # Gradiente vectorizado
        grad = x / 2000 + (1 / sqrt_i) * np.tan(x / sqrt_i) * prod_cos
        
        return grad
        
        return grad

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """
        Return the Hessian matrix at x (2nd derivative)
        
        Diagonal (j = k):
        ∂²f/∂x_j² = 1/2000 + (cos(x_j/√j)/j) * Π_{i≠j}(cos(x_i/√i))
        
        Off-diagonal (j ≠ k):
        ∂²f/∂x_j∂x_k = (sin(x_j/√j)*sin(x_k/√k))/(√j*√k) * Π_{i≠j,i≠k}(cos(x_i/√i))
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        H = np.zeros((n, n))
        
        # Precalcular terminos
        i = np.arange(1, n + 1)
        sqrt_i = np.sqrt(i)
        cos_terms = np.cos(x / sqrt_i)
        sin_terms = np.sin(x / sqrt_i)
        prod_total = np.prod(cos_terms)
        
        for j in range(n):
            # Elemento diagonal H[j,j]
            # 1/2000 + (cos(x_j/√(j+1))/(j+1)) * Π_{i≠j}(cos(x_i/√i))
            
            if cos_terms[j] != 0:
                prod_without_j = prod_total / cos_terms[j]
            else:
                mask = np.ones(n, dtype=bool)
                mask[j] = False
                prod_without_j = np.prod(cos_terms[mask])
            
            H[j, j] = 1/2000 + (cos_terms[j] / (j + 1)) * prod_without_j
            
            # Elementos fuera de la diagonal H[j,k] para k > j
            for k in range(j + 1, n):
                # (sin(x_j/√(j+1))*sin(x_k/√(k+1)))/(√(j+1)*√(k+1)) * Π_{i≠j,i≠k}(cos)
                
                if cos_terms[j] != 0 and cos_terms[k] != 0:
                    prod_without_jk = prod_total / (cos_terms[j] * cos_terms[k])
                else:
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    mask[k] = False
                    prod_without_jk = np.prod(cos_terms[mask])
                
                H[j, k] = (sin_terms[j] * sin_terms[k]) / (sqrt_i[j] * sqrt_i[k]) * prod_without_jk
                H[k, j] = H[j, k]  # Hessiana es simetrica
        
        return H

    def plot_2d(self, range_val=[-10, 10], density=200):
        """
        Grafica la funcion de Griewangk en 2D usando contornos.
        El minimo global está en (0, 0).
        """
        my_plot = Plot(f'Funcion de {self.name}')
        my_plot.canvas(xlabel='x₁', ylabel='x₂')
        my_plot.draw_contours(self, range_val=range_val, density=density)
        my_plot.show()


def main():
    f = Func_Griew()
    
    # Probar en el minimo global
    x_min = np.array([0.0, 0.0])
    print(f"F({x_min}) = {f.eval(x_min)}")  # Debe ser 0
    print(f"∇F({x_min}) = {f.diff(x_min)}")  # Debe ser [0, 0]
    print(f"∇²F({x_min}) =\n{f.ddiff(x_min)}")
    
    print("\n--- Prueba en otro punto ---")
    x_test = np.array([1.0, 2.0])
    print(f"F({x_test}) = {f.eval(x_test)}")
    print(f"∇F({x_test}) = {f.diff(x_test)}")
    print(f"∇²F({x_test}) =\n{f.ddiff(x_test)}")
    
    # Graficar
    f.plot_2d(range_val=[-10, 10], density=200)


if __name__ == "__main__":
    main()
