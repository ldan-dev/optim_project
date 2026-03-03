"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

Para el gradiente, son estos métodos:
* _INIT _ : Constructor que definirá los parámetros a usar, las iteraciones máximas, etc.
* SOLVE(): Lo q implementa el algoritmo.
* PLOT_2D(): que grafique cómo fue la trayectoria a la hora de resolverlo
"""

import matplotlib.pyplot as plt
import numpy as np

from plot import Plot # clase Plot
from functions import * # clases Function


# Importar condiciones de paso y direcciones de descenso
from step_conditions import CONDITIONS as STEP_CONDITIONS
from descent_dir import DIRECTIONS as DESCENT_DIRECTIONS

class GradientDescent():
    """
    Docstring for GradientDescent
    """

    def __init__(self,
                func:Function,
                step_size=1,
                alpha=0.01,
                max_it=10000,
                tolerance=1e-6,
                cond_step="armijo",
                descent_dir='dg'): 
        """
        Constructor que definirá los parámetros a usar, las iteraciones máximas, etc

        - func: the objective function 
        - alpha: learning rate
        - max_it: maximum number of iterations
        - tolerance: avoid infinite loop
        """          

        self.func = func
        self.step_size = step_size
        self.alpha = alpha 
        self.max_it = max_it
        self.tolerance = tolerance
        self.k=0
        self.path = []

        # select the condition step: 
        if cond_step in STEP_CONDITIONS:
            self.cond_step = STEP_CONDITIONS[cond_step]
        else:
            raise ValueError(f"ERROR: step condition NOT FOUND: {cond_step}. Available: {list(STEP_CONDITIONS.keys())}")

        # select the descent direction:
        if descent_dir in DESCENT_DIRECTIONS:
            self.descent_dir = DESCENT_DIRECTIONS[descent_dir]
        else:
            raise ValueError(f"ERROR: descent direction NOT FOUND: {descent_dir}. Available: {list(DESCENT_DIRECTIONS.keys())}")


    # def solve(self, start_point: list, verbose=False):
    #     """ Implementación del algoritmo """
    #     self.path = [] 
    #     xk = np.array(start_point, dtype=float)
    #     self.path.append(xk.copy()) 

    #     for k in range(self.max_it):
    #         grad = self.func.diff(xk) 
    #         xk_next = xk - (self.alpha * grad)
    #         self.path.append(xk_next.copy())
        
    #         if np.linalg.norm(grad) < self.tolerance:
    #             print(f"Converged at iteration {k}") if verbose else print()
    #             break
    #         xk = xk_next
    #     return xk

    def solve(self, start_point: list):
        """ Implementación del algoritmo """
        self.path = [] 
        xk = np.array(start_point, dtype=float)
        self.path.append(xk.copy()) 

        for k in range(self.max_it):
            # 1. Calcular dirección de descenso usando la función seleccionada
            pk = self.descent_dir(self.func, xk)
            
            # 2. Calcular tamaño de paso usando la condición seleccionada
            #    (necesitas implementar una búsqueda de línea que use cond_step)
            alpha = self.line_search(xk, pk)
            
            # 3. Actualizar punto
            xk_next = xk + alpha * pk
            self.path.append(xk_next.copy())
        
            # 4. Criterio de convergencia
            grad = self.func.diff(xk)
            if np.linalg.norm(grad) < self.tolerance:
                print(f"Converged at iteration {k}")
                break
            xk = xk_next
        return xk

    def line_search(self, xk, pk,rho = 0.5 ):
        """
        busqueda de linea usando backtracking con la condición seleccionada.
        Encuentra alpha que satisfaga self.cond_step
        """
        alpha = self.step_size  # empezar con alpha inicial
        # rho = 0.5  # factor de reducción
        
        # reducir alpha hasta que se cumpla la condición
        while not self.cond_step(self.func, xk, alpha, pk):
            alpha *= rho
            if alpha < 1e-10:  # evitar alpha muy pequeño
                break
        return alpha


    def plot_2d(self):
        """  que grafique cómo fue la trayectoria a la hora de resolverlo  puntos con su linea de path"""
        pass

    
def main():
    """  Docstring for main  """




if __name__ == "__main__":
    main()