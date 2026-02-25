"""
Goldstein Condition (Condición de Goldstein)
============================================

La condición de Goldstein proporciona cotas superior e inferior para el
tamaño de paso aceptable, asegurando que el paso no sea ni muy grande ni muy pequeño.

Las condiciones se satisfacen cuando AMBAS se cumplen:
    f(x_k) + (1 - c) * alpha * grad(f(x_k))^T * p_k <= f(x_k + alpha * p_k)
    f(x_k + alpha * p_k) <= f(x_k) + c * alpha * grad(f(x_k))^T * p_k

Lo cual es equivalente a:
    f(x_k) + (1 - c) * alpha * grad(f(x_k))^T * p_k <= f(x_k + alpha * p_k) <= f(x_k) + c * alpha * grad(f(x_k))^T * p_k

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: dirección de descenso
    - c: constante de Goldstein (típicamente c ∈ (0, 0.5), comúnmente c = 0.25)
    - grad(f(x_k)): gradiente de f en x_k

Nota: La segunda desigualdad es equivalente a la condición de Armijo.

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3, página 8
"""

import numpy as np


def goldstein_cond(func, xk, alpha, pk, c=0.25):
    """
    Evalúa la condición de Goldstein para un tamaño de paso dado.

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
    c : float, optional
        Constante de Goldstein, por defecto 0.25.
        Debe estar en (0, 0.5) para que las cotas tengan sentido

    Returns
    -------
    bool
        True si la condición de Goldstein se satisface, False en caso contrario

    Notes
    -----
    A diferencia de las condiciones de Wolfe, Goldstein puede excluir todos los
    minimizadores de la función phi(alpha) = f(x_k + alpha * p_k), lo cual puede
    ser problemático en algunos casos. Por esta razón, las condiciones de Wolfe
    son generalmente preferidas.
    """
    # TODO: Implementar la condición de Goldstein
    pass
