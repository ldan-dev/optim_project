"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 25/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
funcion de griewangk
"""

import numpy as np
# import matplotlib.pyplot as plt
from function import Function
from plot import Plot

class Func_Sphere(Function):
    def __init__(self, limits=[-5, 5]):
        self.limits = limits
        self.path = [] 

    def eval(self, x: np.ndarray) -> float:
        pass

    def diff(self, x: np.ndarray) -> np.ndarray:
        pass

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        # return 2 * np.eye(len(x))
        pass

    def plot2d(self, title="griewangk Function Plot", path=None):
        if path is not None:
            self.path = path

        canvas1 = Plot(title)
        canvas1.canvas()
        
        canvas1.draw_contours(function_obj=self, range_val=self.limits)
        
        canvas1.draw_trace(path_points=self.path)
        canvas1.show()

def main():
    pass

if __name__ == "__main__":
    main()
