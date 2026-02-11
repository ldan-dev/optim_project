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
from function import Function # clase Function

class GradientDescent(object):
    """
    Docstring for GradientDescent
    """

    def __init__(self, func:Function, step_size=1, alpha=0.01, max_it=10000, tolerance=1e-6): 
        """
        Constructor que definirá los parámetros a usar, las iteraciones máximas, etc

        - func: the objective function 
        - alpha: learning rate
        - max_it: maximum number of iterations
        - tolerance: avoid infinite loop
        """
        
        self.func = func
        self.step_size = step_size
        self.alpha = alpha # learning rate
        self.max_it = max_it
        self.tolerance = tolerance
        self.path = [] # points historial for ploting

    def solve(self, start_point:float):
        """  Implementación del algoritmo  """
        
        pass
    
    def plot_2d(self):
        """  que grafique cómo fue la trayectoria a la hora de resolverlo  puntos con su linea de path"""
        pass


def main():
    """  Docstring for main  """



if __name__ == "__main__":
    main()