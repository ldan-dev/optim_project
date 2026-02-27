"""
Newton Direction (Dirección de Newton usando la Hessiana)
=========================================================

La dirección de Newton utiliza información de segundo orden (la matriz Hessiana)
para determinar la dirección de descenso, lo que generalmente resulta en
convergencia más rápida cerca del óptimo.

La dirección se define como:
    p_k = -H(x_k)^{-1} * grad(f(x_k))

Donde:
    - x_k: punto actual
    - H(x_k): matriz Hessiana de f evaluada en x_k (matriz de segundas derivadas)
    - grad(f(x_k)): gradiente de f evaluado en x_k
    - p_k: dirección de descenso resultante

Esto se obtiene resolviendo el sistema lineal:
    H(x_k) * p_k = -grad(f(x_k))

Propiedades:
    - Convergencia cuadrática cerca del óptimo (muy rápida)
    - Requiere que la Hessiana sea definida positiva para ser dirección de descenso
    - Costoso computacionalmente: requiere calcular la Hessiana e invertirla
    - Puede fallar si la Hessiana es singular o no definida positiva

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 2 y 6
"""

import numpy as np


def hessian_dir(func, xk):
    """
    Calcula la dirección de Newton usando la Hessiana.

    Parameters
    ----------
    func : Function
        Objeto función con métodos:
        - diff() para evaluar grad(f(x))
        - hessian() para evaluar H(f(x)) (matriz Hessiana)
    xk : np.ndarray
        Punto actual en el espacio de búsqueda

    Returns
    -------
    np.ndarray
        Dirección de Newton p_k = -H(x_k)^{-1} * grad(f(x_k))

    Raises
    ------
    LinAlgError
        Si la Hessiana es singular y no se puede resolver el sistema

    Notes
    -----
    **IMPORTANTE**: Para que esta dirección sea de descenso, la Hessiana debe
    ser definida positiva. Si no lo es, se pueden usar modificaciones como:
    - Agregar un múltiplo de la identidad: H + mu*I
    - Usar factorización de Cholesky modificada
    - Usar métodos quasi-Newton (BFGS, L-BFGS)
    
    Se recomienda usar np.linalg.solve() en lugar de calcular la inversa
    explícita para mayor estabilidad numérica:
        p_k = np.linalg.solve(H, -grad)
    
    Para funciones cuadráticas f(x) = 0.5 * x^T A x - b^T x, el método
    de Newton converge en una sola iteración.
    """
    # TODO: Implementar la dirección de Newton
    # Hint: usar np.linalg.solve(H, -grad) para resolver H * p = -grad
    pass
