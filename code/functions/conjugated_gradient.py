import matplotlib.pyplot as plt
import numpy as np

from plot import Plot 
from functions import *
from gradient_descend import GradientDescent

class ConjugateGradient(GradientDescent):
    def __init__(self, func, max_it, tolerance=1e-6):
        super().__init__(func, alpha=0, max_it=max_it)
        self.tolerance = tolerance
        self.path = []

    def solve(self, start_point: list, verbose=False):
        self.path = []
        xo = np.array(start_point, dtype=float)
        self.path.append(xo.copy())
        g = self.func.diff(xo)
        p = -g
        k = 0
        while np.linalg.norm(g) > self.tolerance and k < self.max_it:
            A_mx = self.func.ddiff(xo)
            Apk = A_mx @ p
            denom_alpha = p.T @ Apk
            if abs(denom_alpha) < 1e-15: 
                break 
            alpha = (g.T @ g) / denom_alpha
            xo = xo + alpha * p
            self.path.append(xo.copy())
            g_next = self.func.diff(xo)
            beta = (g_next.T @ g_next) / (g.T @ g)
            p_next = -g_next + beta * p
            g = g_next
            p = p_next
            k = k + 1
            if verbose:
                print(f"Iteracion {k}: Error (Norma g) = {np.linalg.norm(g)}")
                
        return xo
if __name__ == "__main__":
    
    print(f"Vector b esperado: {f_test.b}")
