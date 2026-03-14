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
    class FuncSuavizado:
        def __init__(self, b_ruidoso):
            self.b = np.array(b_ruidoso, dtype=float)
            n = len(self.b)
            # Matriz A que penaliza cambios bruscos (Suavizado)
            self.A = np.eye(n) * 2
            for i in range(n-1):
                self.A[i, i+1] = self.A[i+1, i] = -1

        def diff(self, x): 
            return (self.A @ x) - self.b
            
        def ddiff(self, x): 
            return self.A

    # 1. Pixeles con un "salto" y ruido
    # Imagina que la imagen deberia ser [0,0,0,10,10,10]
    pixeles_ruido = [0.1, -0.2, 0.5, 9.8, 10.2, 9.5]
    
    f_img = FuncSuavizado(pixeles_ruido)
    cg = ConjugateGradient(func=f_img, max_it=20)
    
    # 2. Resolver
    suave = cg.solve(start_point=pixeles_ruido, verbose=True)
    
    print("\n--- PRUEBA DE SUAVIZADO DE PIXELES ---")
    print(f"Originales (con ruido): {pixeles_ruido}")
    print(f"Suavizados (resultado): {np.round(suave, 2)}")