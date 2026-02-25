"""
Armijo Condition (Condición de Armijo)
======================================

La condición de Armijo, también conocida como condición de descenso suficiente,
asegura que el tamaño de paso produce una reducción suficiente en la función objetivo.

La condición se satisface cuando:
    f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * grad(f(x_k))^T * p_k

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: dirección de descenso
    - c1: constante de Armijo (típicamente c1 ∈ (0, 1), comúnmente c1 = 1e-4)
    - grad(f(x_k)): gradiente de f en x_k

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3
"""

import numpy as np


def armijo_cond(func, xk, alpha, pk, c1=1e-4):
    """
    Evalúa la condición de Armijo para un tamaño de paso dado.

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
        Constante de Armijo, por defecto 1e-4. Debe estar en (0, 1)

    Returns
    -------
    bool
        True si la condición de Armijo se satisface, False en caso contrario

    Notes
    -----
    La condición de Armijo por sí sola puede aceptar pasos muy pequeños.
    Por eso generalmente se combina con otras condiciones (como Wolfe curvature).
    """
    # TODO: Implementar la condición de Armijo
    pass
