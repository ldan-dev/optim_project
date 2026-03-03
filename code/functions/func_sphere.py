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
try:
    from .function import Function
except ImportError:
    from function import Function
from plot import Plot
from gradient_descend import GradientDescent

class Func_Sphere(Function):
    def __init__(self, limits=[-5, 5]):
        super().__init__()
        self.limits = limits
        self.name = "Esfera"

    def eval(self, x: np.ndarray) -> float:
        x = self._Function__validate_x(x)
        return np.sum(x**2)

    def diff(self, x: np.ndarray) -> np.ndarray:
        x = self._Function__validate_x(x)
        return 2 * x

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        x = self._Function__validate_x(x)
        return 2 * np.eye(len(x))

def main():
    esfera = Func_Sphere(limits=[-10, 10])
    gd_esfera = GradientDescent(func=esfera, alpha=0.1, max_it=100)
    gd_esfera.solve(start_point=[8.0, 8.0])
    gd_esfera.plot_2d()
if __name__ == "__main__":
    main()
