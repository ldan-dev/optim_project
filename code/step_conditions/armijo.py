"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

Armijo Condition (Condicion de Armijo)

La condicion de Armijo, tambien conocida como condicion de descenso suficiente,
asegura que el tamaño de paso produce una reduccion suficiente en la funcion objetivo.

La condicion se satisface cuando:
    f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * grad(f(x_k))^T * p_k

Donde:
    - x_k: punto actual
    - alpha: tamaño de paso (step size)
    - p_k: direccion de descenso
    - c1: constante de Armijo (tipicamente c1 ∈ (0, 1), comúnmente c1 = 1e-4)
    - grad(f(x_k)): gradiente de f en x_k

Referencias:
    - Nocedal & Wright, "Numerical Optimization", Chapter 3
"""

import numpy as np
from function import Function

def armijo_cond(func:Function, xk, alpha: float, pk: np.ndarray, c1=1e-4):
    """
    Evalua la condicion de Armijo para un tamaño de paso dado.

    func : objective Function
    xk : punto actual en el espacio de búsqueda
    alpha : tamaño de paso propuesto (alpha > 0)
    pk : direccion de descenso (debe satisfacer grad(f(x_k))^T * p_k < 0)
    c1 : constante de Armijo, por defecto 1e-4. Debe estar en (0, 1)

    Returna
        True si la condicion de Armijo se satisface, False en caso contrario

    La condicion de Armijo por si sola puede aceptar pasos muy pequeños
    Por eso generalmente se combina con otras condiciones (como Wolfe curvature).
    """
    fk = func.eval(xk)                  # f(x_k)
    fk_new = func.eval(xk + alpha * pk) # f(x_k + alpha * p_k)
    grad_k = func.diff(xk)              # grad(f(x_k))
    
    # lado derecho: f(x_k) + c1 * alpha * grad^T * p
    rhs = fk + c1 * alpha * np.dot(grad_k, pk)
    
    return fk_new <= rhs
