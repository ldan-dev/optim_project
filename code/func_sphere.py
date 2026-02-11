"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""
import numpy as np
import matplotlib.pyplot as plt
from function import Function

class Func_Sphere(Function):
    """
    Docstring for Func_Sphere
    """
    def __init__(self):
        pass
    def eval(self, x: np.ndarray) -> float:
        return np.sum(x**2)
    def diff(self, x: np.ndarray) -> np.ndarray:
        return 2 * np.array(x)

def main():
    """  Docstring for main  """


# functions here


if __name__ == "__main__":
    main()
