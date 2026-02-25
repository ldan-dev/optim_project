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
        
        f(x) = SUM(x_i²/4000) - MULTIPLICATORIA(cos(x_i/√i)) + 1
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        # Termino de suma
        sum_term = np.sum(x**2) / 4000
        
        # Termino de producto: MULTI(cos(x_i/√i))
        # i va de 1 a n (indice basado en 1)
        i = np.arange(1, n + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        
        return sum_term - prod_term + 1

    def diff(self, x: np.ndarray) -> np.ndarray:
        """
        Return the gradient vector at x (1st derivative)
        
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        i = np.arange(1, n + 1)
        sqrt_i = np.sqrt(i)
        
        # producto completo de todos los cosenos
        prod_cos = np.prod(np.cos(x / sqrt_i))
        
        # gradiente vectorizado
        grad = x / 2000 + (1 / sqrt_i) * np.tan(x / sqrt_i) * prod_cos
        
        return grad
        

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """
        Return the Hessian matrix at x (2nd derivative)
        """
    def ddiff(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        i = np.arange(1, n + 1)
        sqrt_i = np.sqrt(i)
        
        prod_cos = np.prod(np.cos(x / sqrt_i))
        tan_terms = np.tan(x / sqrt_i)
        
        # diagonal: 1/2000 + (1/j) * prod_cos
        diag = 1/2000 + prod_cos / i
        
        # Off-diagonal con outer products
        H = -prod_cos * np.outer(tan_terms, tan_terms) / np.outer(sqrt_i, sqrt_i)
        np.fill_diagonal(H, diag)
        
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
    
    # en el minimo global
    x_min = np.array([0.0, 0.0])
    print(f"F({x_min}) = {f.eval(x_min)}")  # debe ser 0
    print(f"∇F({x_min}) = {f.diff(x_min)}")  # debe ser [0, 0]
    print(f"∇²F({x_min}) =\n{f.ddiff(x_min)}")
    
    print("\n--- prueba en otro punto ---")
    x_test = np.array([1.0, 2.0])
    print(f"F({x_test}) = {f.eval(x_test)}")
    print(f"∇F({x_test}) = {f.diff(x_test)}")
    print(f"∇²F({x_test}) =\n{f.ddiff(x_test)}")
    
    f.plot_2d()


if __name__ == "__main__":
    main()
