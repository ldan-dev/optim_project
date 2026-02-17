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
from plot import Plot

class Func_Sphere(Function):
    """
    Docstring for Func_Sphere
    """
    def __init__(self, limits=[-5, 5]):
        self.limits = limits
        self.path = []
    def eval(self, x: np.ndarray) -> float:
        return np.sum(x**2)
    def diff(self, x: np.ndarray) -> np.ndarray:
        return 2 * np.array(x)
    def ddiff(self, x: np.ndarray) -> np.ndarray:
        return 2 * np.eye(len(x)) #matriz identidad de len(x)

def main():
    """  Docstring for main  """
    canvas1 = Plot(title="Gráfica de la Esfera")
    canvas1.canvas()

    f1 = Func_Sphere(limits=[-10, 10])
    f1.path = [[-8, 8], [-4, 4], [-2, 2], [0, 0]]

    canvas1.draw_contours(function_obj=f1, range_val=f1.limits)

    canvas1.draw_trace(path_points=f1.path)
    canvas1.show()
        
if __name__ == "__main__":
    main()
