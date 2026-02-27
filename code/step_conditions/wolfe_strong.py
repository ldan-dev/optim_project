"""
Strong Wolfe Conditions (Condiciones Fuertes de Wolfe)
======================================================

Las condiciones fuertes de Wolfe combinan la condición de descenso suficiente
con una versión más restrictiva de la condición de curvatura que usa valor absoluto.

Las condiciones se satisfacen cuando AMBAS se cumplen:
    1) f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * grad(f(x_k))^T * p_k
    2) |grad(f(x_k + alpha * p_k))^T * p_k| <= c2 * |grad(f(x_k))^T * p_k|

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: dirección de descenso
    - c1: constante de descenso suficiente (típicamente c1 = 1e-4)
    - c2: constante de curvatura (típicamente c2 ∈ (c1, 1), comúnmente c2 = 0.9)
    - grad(f(x)): gradiente de f en x

La diferencia con las condiciones de Wolfe estándar es el valor absoluto en la
segunda condición, lo que excluye puntos donde la derivada direccional es muy
negativa o muy positiva.

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3, página 7
"""

import numpy as np


def wolfe_strong_cond(func, xk, alpha, pk, c1=1e-4, c2=0.9):
    """
    Evalúa las condiciones fuertes de Wolfe.

    Parameters
    ----------
    func : Function
        Objeto función con métodos eval() y diff() para evaluar f(x) y grad(f(x))
    xk : np.ndarray
        Punto actual en el espacio de búsqueda
    alpha : float
        Tamaño de paso propuesto (alpha > 0)
    pk : np.ndarray
        Dirección de descenso (debe satisfacer grad(f(x_k))^T * p_k < 0)
    c1 : float, optional
        Constante de descenso suficiente, por defecto 1e-4.
        Debe estar en (0, 1) y c1 < c2
    c2 : float, optional
        Constante de curvatura, por defecto 0.9.
        Debe estar en (c1, 1)

    Returns
    -------
    bool
        True si AMBAS condiciones fuertes de Wolfe se satisfacen, False en caso contrario

    Notes
    -----
    Las condiciones fuertes de Wolfe son más restrictivas que las condiciones
    de Wolfe estándar y son preferidas para métodos quasi-Newton como BFGS.
    """
    # TODO: Implementar las condiciones fuertes de Wolfe
    pass
