"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

Newton Direction (Direccion de Newton usando la Hessiana)

La direccion de Newton utiliza informacion de segundo orden (la matriz Hessiana)
para determinar la direccion de descenso, lo que generalmente resulta en
convergencia mas rapida cerca del optimo.

La direccion se define como:
    p_k = -H(x_k)^{-1} * grad(f(x_k))

Donde:
    - x_k: punto actual
    - H(x_k): matriz Hessiana de f evaluada en x_k (matriz de segundas derivadas)
    - grad(f(x_k)): gradiente de f evaluado en x_k
    - p_k: direccion de descenso resultante

Esto se obtiene resolviendo el sistema lineal:
    H(x_k) * p_k = -grad(f(x_k))

    parecido a:
    Ax = B

Propiedades:
    - Convergencia cuadratica cerca del optimo (muy rapida)
    - Requiere que la Hessiana sea definida positiva para ser direccion de descenso
    - Costoso computacionalmente: requiere calcular la Hessiana e invertirla
    - Puede fallar si la Hessiana es singular o no definida positiva

    
Se recomienda usar np.linalg.solve() en lugar de calcular la inversa
explicita para mayor estabilidad numerica:
    p_k = np.linalg.solve(H, -grad)

Para funciones cuadraticas f(x) = 0.5 * x^T A x - b^T x, el metodo
de Newton converge en una sola iteracion.

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 2 y 6
"""

import numpy as np
from function import Function

def hessian_dir(func:Function, xk:np.ndarray) -> np.ndarray:
    """
    Calcula la direccion de Newton usando la Hessiana.

    Parameters
    ----------
    func : Function
        Objeto funcion con metodos:
        - diff() para evaluar grad(f(x))
        - hessian() para evaluar H(f(x)) (matriz Hessiana)
    xk : np.ndarray
        Punto actual en el espacio de busqueda

    Direccion de Newton p_k = -H(x_k)^{-1} * grad(f(x_k))

    LinAlgError
        Si la Hessiana es singular y no se puede resolver el sistema
    
    """
    grad = func.diff(xk)    # gradiente
    H = func.ddiff(xk)      # hessiana
    
    # H * p = -grad
    pk = np.linalg.solve(H, -grad)
    
    return pk
