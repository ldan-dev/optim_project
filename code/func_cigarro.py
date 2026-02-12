"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""
from function import Function
import numpy as np
class Func_Cigarro(Function):
    """
    Docstring for Func_Cigarro
    """
    def __init__(self, params):
        pass
    def eval(self, x: np.ndarray) -> float:
        return np.sum(x**2) + 1000000 * np.sum(x[1:]**2)
    def diff(self, x: np.ndarray) -> np.ndarray:
        if x == 1:
            return 2 * x
        else:
            return 2000000 * x
    def ddiff(self, x: np.ndarray) -> np.ndarray:
        if x == 1:
            return 2
        else: 
            return 2000000
        
    

def main():
    """  Docstring for main  """


# functions here


if __name__ == "__main__":
    main()