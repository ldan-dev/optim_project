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

    def __init__(self, func:Function, alpha=0.01, max_it=10000, tolerance=1e-6): 
        """
        Constructor que definirá los parámetros a usar, las iteraciones máximas, etc

        - func: the objective function 
        - alpha: learning rate
        - max_it: maximum number of iterations
        - tolerance: 
        """
        
        self.func = func
        self.alpha = alpha
        self.max_it = max_it
        self.tolerance = tolerance

    def solve():
        """  Implementación del algoritmo  """
        pass
    
    def plot_2d():
        """  que grafique cómo fue la trayectoria a la hora de resolverlo  """
        pass


def main():
    """  Docstring for main  """


# functions here


if __name__ == "__main__":
    main()