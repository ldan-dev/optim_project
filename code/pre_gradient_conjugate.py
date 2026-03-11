import matplotlib.pyplot as plt
import numpy as np

from plot import Plot # clase Plot
from functions import * # clases Function

from step_conditions import CONDITIONS as STEP_CONDITIONS
from descent_dir import DIRECTIONS as DESCENT_DIRECTIONS

import numpy as np

class Pre_Gradient_Conjugate:
    def __init__(self, func, max_it=5000, tolerance=1e-6):
        self.func = func
        self.max_it = max_it
        self.tolerance = tolerance
        self.path = []

    def get_preconditioner_inv(self, x):
        """ 
        Implementación de Jacobi Preconditioning: M = diag(Hessian)
        Retornamos la inversa de M para facilitar el cálculo y_k = M^-1 * g_k
        """
        H = self.func.ddiff(x)
        diag = np.diag(H)
        # Evitamos división por cero
        diag_inv = np.array([1/d if abs(d) > 1e-12 else 1.0 for d in diag])
        return np.diag(diag_inv)

    def solve(self, start_point: list):
        self.path = []
        xk = np.array(start_point, dtype=float)
        self.path.append(xk.copy())

        # 1. Inicializar: g0 = gradiente, M^-1, y0, p0
        gk = self.func.diff(xk)
        M_inv = self.get_preconditioner_inv(xk)
        yk = M_inv @ gk
        pk = -yk  # Dirección inicial 
        for k in range(self.max_it):
            A = self.func.ddiff(xk)
            Apk = A @ pk
            
            # 2. Calcular alpha_k (Tamaño de paso óptimo para cuadráticas)
            # Denominador: pk^T * A * pk
            denom = pk.T @ Apk
            if abs(denom) < 1e-15: break
                
            # Numerador: gk^T * yk 
            alpha_k = (gk.T @ yk) / denom
            
            # 3. Actualizar punto y residuo (gradiente)
            xk_next = xk + alpha_k * pk
            self.path.append(xk_next.copy())
            
            gk_next = gk + alpha_k * Apk
            
            # 4. Criterio de parada
            if np.linalg.norm(gk_next) < self.tolerance:
                print(f"PCG converged at iteration {k}")
                break
                
            # 5. Precondicionamiento para el siguiente paso
            M_inv = self.get_preconditioner_inv(xk_next)
            yk_next = M_inv @ gk_next
            
            # 6. Calcular Beta 
            beta_k = (gk_next.T @ yk_next) / (gk.T @ yk)
            
            # 7. Nueva dirección de búsqueda
            pk = -yk_next + beta_k * pk
            
            # Actualizar variables para la siguiente iteración
            xk, gk, yk = xk_next, gk_next, yk_next

        return xk