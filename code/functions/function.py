"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plot import Plot # clase Plot

class Function():
    """
    Docstring for Function
    """
    def __init__(self, params=None):
        self.params = params

    def __validate_x(self, x: np.ndarray) -> np.ndarray:
        """Valida y convierte x a vector numpy 1D numérico."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a 1D vector")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("x must have numerical elements")
        return x

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function"""
        
        x = np.asarray(x) # convert to array
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("x must have numerical elements")
        
        # todo: implement function

    def diff(self, x: np.ndarray) -> np.ndarray:
        """  1st derivate  """
        raise NotImplementedError("diff method must be implemented in subclass")

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """  2nd derivate  """
        raise NotImplementedError("ddiff method must be implemented in subclass")

    def plot_2d(self, lim:list[float], canva:Plot ):
        """  plot the function:  """
        pass



def main():
    """  Docstring for main  """


if __name__ == "__main__":
    main()