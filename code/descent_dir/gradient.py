"""
Negative Gradient Direction (Dirección del Gradiente Negativo)
==============================================================

La dirección de descenso más simple es el gradiente negativo de la función
objetivo. Esta es la dirección de máximo descenso local.

La dirección se define como:
    p_k = -grad(f(x_k))

Donde:
    - x_k: punto actual
    - grad(f(x_k)): gradiente de f evaluado en x_k
    - p_k: dirección de descenso resultante

Propiedades:
    - Siempre es una dirección de descenso (si grad(f(x_k)) != 0)
    - Es la dirección de máxima pendiente negativa
    - Convergencia puede ser lenta para funciones mal condicionadas

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 2
"""

import numpy as np


def gradient_dir(func, xk):
    """
    Calcula la dirección de descenso como el gradiente negativo.

    Parameters
    ----------
    func : Function
        Objeto función con método diff() para evaluar grad(f(x))
    xk : np.ndarray
        Punto actual en el espacio de búsqueda

    Returns
    -------
    np.ndarray
        Dirección de descenso p_k = -grad(f(x_k))

    Notes
    -----
    Esta es la dirección más básica de descenso. Es simple pero puede
    resultar en convergencia lenta para funciones con alto número de condición
    (eigenvalores de la Hessiana muy diferentes entre sí).
    
    Para funciones cuadráticas f(x) = 0.5 * x^T A x - b^T x, el método
    de gradiente puede requerir muchas iteraciones si los eigenvalores
    de A tienen gran dispersión.
    """
    # TODO: Implementar la dirección del gradiente negativo
    pass
