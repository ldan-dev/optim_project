"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""
from plot import Plot
import numpy as np
from function import Function

class Func_Cigarro(Function):
    def __init__(self, limits=[-20, 10]):
        self.limits = limits
        self.path = []

    def eval(self, x: np.ndarray) -> float:
        val_real = x[0]**2 + 1_000_000 * np.sum(x[1:]**2)
        return np.log10(val_real + 1)

    def diff(self, x: np.ndarray) -> np.ndarray:

        multiplicadores = np.full_like(x, 2_000_000, dtype=float)
        multiplicadores[0] = 2.0
        return multiplicadores * x

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        hess_diag = np.full_like(x, 2_000_000, dtype=float)
        hess_diag[0] = 2.0
        return hess_diag

def main():
    """  Docstring for main  """
    canvas1 = Plot(title="Función Cigarro")
    canvas1.canvas()

    func_cigarro = Func_Cigarro(limits=[-10,10])
    func_cigarro.path = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]

    canvas1.draw_contours(function_obj=func_cigarro, range_val = func_cigarro.limits)

    canvas1.draw_trace(path_points = func_cigarro.path)
    canvas1.show()
# functions here


if __name__ == "__main__":
    main()