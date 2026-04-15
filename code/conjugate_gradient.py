"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 13/03/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA:
DESCRIPTION:
Conjugate Gradient Method (vanilla y preconditioned)
para resolver sistemas SPD del tipo:

    A x = b

en este proyecto, típicamente provenientes de una función cuadrática.
"""

import numpy as np


class ConjugateGradientSolver:
    def __init__(self, max_it: int = 200, tolerance: float = 1e-6):
        self.max_it = max_it
        self.tolerance = tolerance
        self.path = []
        self.residual_norms = []

    def _run_cg(self, apply_A, b, x0, preconditioner=None):
        xk = np.asarray(x0, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)

        if xk.size != b.size:
            raise ValueError("x0 y b deben tener la misma dimensión")

        gk = apply_A(xk) - b  # residuo del gradiente
        if preconditioner is None:
            zk = gk.copy()
        else:
            zk = preconditioner(gk)

        pk = -zk

        self.path = [xk.copy()]
        self.residual_norms = [np.linalg.norm(gk)]

        if self.residual_norms[-1] < self.tolerance:
            return xk

        for _ in range(self.max_it):
            Apk = apply_A(pk)
            denom = float(np.dot(pk, Apk))
            if abs(denom) < 1e-20:
                break

            if preconditioner is None:
                num = float(np.dot(gk, gk))
            else:
                num = float(np.dot(gk, zk))

            alpha = num / denom
            xk_new = xk + alpha * pk
            gk_new = gk + alpha * Apk

            self.path.append(xk_new.copy())
            self.residual_norms.append(np.linalg.norm(gk_new))

            if self.residual_norms[-1] < self.tolerance:
                xk = xk_new
                break

            if preconditioner is None:
                beta = float(np.dot(gk_new, gk_new)) / max(float(np.dot(gk, gk)), 1e-20)
                pk = -gk_new + beta * pk
            else:
                zk_new = preconditioner(gk_new)
                beta = float(np.dot(gk_new, zk_new)) / max(float(np.dot(gk, zk)), 1e-20)
                pk = -zk_new + beta * pk
                zk = zk_new

            xk, gk = xk_new, gk_new

        return xk

    def solve_vanilla(self, apply_A, b, x0):
        """Conjugate Gradient estándar (sin precondicionador)."""
        return self._run_cg(apply_A=apply_A, b=b, x0=x0, preconditioner=None)

    def solve_preconditioned(self, apply_A, b, x0, M_diag):
        """
        Preconditioned CG con precondicionador de Jacobi:
            M = diag(A)
        """
        M_diag = np.asarray(M_diag, dtype=float).reshape(-1)

        def jacobi_preconditioner(g):
            return g / np.maximum(M_diag, 1e-12)

        return self._run_cg(apply_A=apply_A, b=b, x0=x0, preconditioner=jacobi_preconditioner)
