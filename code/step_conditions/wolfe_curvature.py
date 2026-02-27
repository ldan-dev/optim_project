"""
Wolfe Curvature Condition (Condición de Curvatura de Wolfe)
===========================================================

La condición de curvatura de Wolfe (segunda condición de Wolfe) asegura que
el tamaño de paso no sea demasiado pequeño, requiriendo que la pendiente
en el nuevo punto sea suficientemente mayor que en el punto actual.

La condición se satisface cuando:
    grad(f(x_k + alpha * p_k))^T * p_k >= c2 * grad(f(x_k))^T * p_k

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: dirección de descenso
    - c2: constante de curvatura (típicamente c2 ∈ (c1, 1), comúnmente c2 = 0.9)
    - grad(f(x)): gradiente de f en x

Esta condición evita que el algoritmo acepte pasos muy pequeños que
satisfacen Armijo pero no hacen progreso significativo.

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3, página 6
"""

import numpy as np


def wolfe_curvature_cond(func, xk, alpha, pk, c2=0.9):
    """
    Evalúa la condición de curvatura de Wolfe.

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
    c2 : float, optional
        Constante de curvatura de Wolfe, por defecto 0.9.
        Debe estar en (c1, 1) donde c1 es la constante de descenso suficiente

    Returns
    -------
    bool
        True si la condición de curvatura se satisface, False en caso contrario

    Notes
    -----
    Esta condición por sí sola no es suficiente; debe combinarse con la 
    condición de descenso suficiente (Armijo/Wolfe descent) para formar
    las condiciones de Wolfe completas.
    """
    

    # 1. Calcular el nuevo punto sumando el paso en la dirección p_k
    x_next = xk + (alpha * pk)
    
    # 2. Obtener los gradientes en el punto actual y en el nuevo punto
    grad_xk = func.diff(xk)
    grad_next = func.diff(x_next)
    
    # 3. Calcular la derivada direccional en ambos puntos (producto punto)
    # Lado izquierdo (derivada direccional en el nuevo punto)
    lhs = np.dot(grad_next, pk)
    
    # Lado derecho (c2 * derivada direccional en el punto actual)
    rhs = c2 * np.dot(grad_xk, pk)
    
    # 4. Verificamos si se cumple la condición
    return bool(lhs >= rhs)
