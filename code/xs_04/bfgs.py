"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""

import os
import sys
import numpy as np

# Permite ejecutar el módulo directo desde /code
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from step_conditions import CONDITIONS
from descent_dir import DIRECTIONS


class BFGSOptimizer:
    """
    Esqueleto base de BFGS:
    - Usa `Function.eval` y `Function.diff`
    - Usa `step_conditions` para búsqueda de línea
    - Puede usar `descent_dir` como fallback inicial
    """

    def __init__(
        self,
        func,
        max_it: int = 100,
        tolerance: float = 1e-6,
        cond_step: str = "wolfe_strong",
        descent_dir: str = "gradient",
        alpha0: float = 1.0,
        rho: float = 0.5,
    ):
        self.func = func
        self.max_it = max_it
        self.tolerance = tolerance
        self.alpha0 = alpha0
        self.rho = rho

        if cond_step not in CONDITIONS:
            raise ValueError(f"step condition no encontrada: {cond_step}")
        self.cond_step = CONDITIONS[cond_step]

        if descent_dir not in DIRECTIONS:
            raise ValueError(f"descent direction no encontrada: {descent_dir}")
        self.descent_dir = DIRECTIONS[descent_dir]

        self.path = []
        self.grad_norms = []

    def _line_search(self, xk: np.ndarray, pk: np.ndarray) -> float:
        alpha = self.alpha0
        while not self.cond_step(self.func, xk, alpha, pk):
            alpha *= self.rho
            if alpha < 1e-12:
                break
        return alpha

    def solve(self, x0: np.ndarray) -> np.ndarray:
        xk = np.asarray(x0, dtype=float).reshape(-1)
        n = xk.size
        Hk = np.eye(n, dtype=float)

        self.path = [xk.copy()]
        self.grad_norms = []

        for _ in range(self.max_it):
            gk = self.func.diff(xk)
            gnorm = np.linalg.norm(gk)
            self.grad_norms.append(float(gnorm))
            if gnorm < self.tolerance:
                break

            # BFGS: p_k = -H_k g_k; fallback a gradiente si es necesario.
            pk = -Hk @ gk
            if np.dot(pk, gk) >= 0:
                pk = self.descent_dir(self.func, xk)

            alpha = self._line_search(xk, pk)
            x_next = xk + alpha * pk

            sk = x_next - xk
            g_next = self.func.diff(x_next)
            yk = g_next - gk

            ys = float(np.dot(yk, sk))
            if ys > 1e-12:
                rho_k = 1.0 / ys
                I = np.eye(n)
                # Update BFGS de la inversa Hessiana.
                Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

            xk = x_next
            self.path.append(xk.copy())

        return xk


def main():
    """Base mínima: el uso completo está en xs04_main.py."""
    print("BFGS base listo.")


if __name__ == "__main__":
    main()