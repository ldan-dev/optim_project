"""
Step Conditions Module
======================

Este módulo contiene las diferentes condiciones de paso (step conditions)
para el algoritmo de gradient descent.

Condiciones disponibles:
- armijo: Condición de Armijo (sufficient decrease)
- wolfe_descent: Condición de Wolfe de descenso suficiente
- wolfe_curvature: Condición de curvatura de Wolfe  
- wolfe_strong: Condiciones fuertes de Wolfe
- goldstein: Condición de Goldstein
"""

from .armijo import armijo_cond
from .wolfe_descent import wolfe_descent_cond
from .wolfe_curvature import wolfe_curvature_cond
from .wolfe_strong import wolfe_strong_cond
from .goldstein import goldstein_cond

# Diccionario para selección de condiciones por nombre
CONDITIONS = {
    "armijo": armijo_cond,
    "wolfe_descent": wolfe_descent_cond,
    "wolfe_curvature": wolfe_curvature_cond,
    "wolfe_strong": wolfe_strong_cond,
    "goldstein": goldstein_cond,
}

__all__ = [
    "armijo_cond",
    "wolfe_descent_cond", 
    "wolfe_curvature_cond",
    "wolfe_strong_cond",
    "goldstein_cond",
    "CONDITIONS",
]
