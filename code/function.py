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
    Docstring for Function
    """
    def __init__(self, params):
        pass

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function"""
        
        x = np.asarray(x) # convert to array
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("x must have numerical elements")
        
        # todo: implement function

    def diff(self) -> np.ndarray:
        """  1st derivate  """
        pass

    def ddiff(self) -> np.ndarray:
        """  2nd derivate  """
        pass

    def plot_2d(self):
        """  plot the function  """
        pass



def main():
    """  Docstring for main  """


if __name__ == "__main__":
    main()