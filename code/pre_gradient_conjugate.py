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
        diag_H = np.diag(H) if H.ndim > 1 else H 
        
        diag_inv = 1.0 / diag_H
        return np.diag(diag_inv)

    def solve(self, start_point: list):
        self.path = []
        xk = np.array(start_point, dtype=float)
        self.path.append(xk.copy())

        gk = self.func.diff(xk)
        if np.linalg.norm(gk) < self.tolerance:
            return xk

        M_inv = self.get_preconditioner_inv(xk)
        yk = M_inv @ gk
        pk = -yk

        for k in range(self.max_it):
            A = self.func.ddiff(xk)
            Apk = A @ pk
            
            # Alfa: (gk^T * yk) / (pk^T * A * pk)
            denom = pk.T @ Apk
            alpha_k = (gk.T @ yk) / denom
            
            xk_next = xk + alpha_k * pk
            self.path.append(xk_next.copy())
            
            gk_next = self.func.diff(xk_next) 
            
            if np.linalg.norm(gk_next) < self.tolerance:
                print(f"PCG converged at iteration {k}")
                xk = xk_next 
                break
            
            M_inv_next = self.get_preconditioner_inv(xk_next)
            yk_next = M_inv_next @ gk_next
            beta_k = (gk_next.T @ yk_next) / (gk.T @ yk)
            
            pk = -yk_next + beta_k * pk
            xk, gk, yk = xk_next, gk_next, yk_next

        return xk