"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

"""

import numpy as np
from plot import Plot # clase Plot

class Function():
    """
    Function abstract class
    """
    def __init__(self, params):
        pass

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function"""
        
        x = self.__validate_x(x)
        
        # todo: implement function

    def diff(self) -> np.ndarray:
        """  1st derivate  """
        pass

    def ddiff(self) -> np.ndarray:
        """  2nd derivate  """
        pass

    def plot_2d(self, lim:list[float], canva:Plot ):
        """  plot the function:  """
        pass

    def __validate_x(self, x):
        x = np.asarray(x, dtype=float) # convert to array
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("x must have numerical elements")
        
        return x


def main():
    """  Docstring for main  """


if __name__ == "__main__":
    main()