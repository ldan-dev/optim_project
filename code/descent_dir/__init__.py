"""
Descent Directions Module
=========================

Este módulo contiene las diferentes direcciones de descenso
para el algoritmo de gradient descent.

Direcciones disponibles:
- gradient: Dirección del gradiente negativo (-grad f(x))
- hessian: Dirección de Newton usando la Hessiana
"""

from .gradient import gradient_dir
from .hessian import hessian_dir

# Diccionario para selección de direcciones por nombre
DIRECTIONS = {
    "dg": gradient_dir,      # dg = descent gradient
    "gradient": gradient_dir,
    "newton": hessian_dir,
    "hessian": hessian_dir,
}

__all__ = [
    "gradient_dir",
    "hessian_dir",
    "DIRECTIONS",
]
