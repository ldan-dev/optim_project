"""
Wolfe Sufficient Decrease Condition (Condición de Descenso Suficiente de Wolfe)
===============================================================================

La condición de descenso suficiente de Wolfe (también conocida como primera 
condición de Wolfe) es equivalente a la condición de Armijo.

La condición se satisface cuando:
    f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * grad(f(x_k))^T * p_k

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: dirección de descenso  
    - c1: constante (típicamente c1 ∈ (0, 1), comúnmente c1 = 1e-4)
    - grad(f(x_k)): gradiente de f en x_k

Esta condición asegura que la función decrece "suficientemente" en cada iteración.

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3, página 5
"""

import numpy as np


def wolfe_descent_cond(func, xk, alpha, pk, c1=1e-4):
    """
    Evalúa la condición de descenso suficiente de Wolfe.

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
        Constante de Wolfe para descenso suficiente, por defecto 1e-4.
        Debe estar en (0, 1) y típicamente c1 < c2 (donde c2 es la constante de curvatura)

    Returns
    -------
    bool
        True si la condición se satisface, False en caso contrario

    Notes
    -----
    Esta condición es idéntica a la condición de Armijo. Se presenta por separado
    para claridad cuando se usa junto con la condición de curvatura de Wolfe.
    """
    # TODO: Implementar la condición de descenso suficiente de Wolfe
    pass
