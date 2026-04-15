"""Módulos base para registro de imágenes afín (no rígido)."""

from .modelo import AffineModel6
from .func_obj import AffineRegistrationObjective
from .bfgs import BFGSOptimizer

__all__ = [
    "AffineModel6",
    "AffineRegistrationObjective",
    "BFGSOptimizer",
]
