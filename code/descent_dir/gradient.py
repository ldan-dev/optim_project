"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 25/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: Optimizacion
DESCRIPTION: 
La direccion se define como:
    p_k = -grad(f(x_k))

Donde:
    - x_k: punto actual
    - grad(f(x_k)): gradiente de f evaluado en x_k
    - p_k: direccion de descenso resultante

Propiedades:
    - Siempre es una direccion de descenso (si grad(f(x_k)) != 0)
    - Es la direccion de máxima pendiente negativa
    - Convergencia puede ser lenta para funciones mal condicionadas
"""

import numpy as np
# from function import Function  # Solo para type hints

def gradient_dir(func, xk:np.ndarray) -> np.ndarray:
    """
    Calcula la direccion de descenso como el gradiente negativo.

    func : objetive funcion con metodo diff() para evaluar grad(f(x))
    xk : punto actual en el espacio de busqueda
    """
    return -func.diff(xk) # p_k = -grad(f(x_k))
