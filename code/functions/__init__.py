"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 03/03/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 

"""

from .function import Function
from .func_sphere import Func_Sphere
from .func_rosen import Func_Rosen
from .func_griew import Func_Griew
from .func_cigarro import Func_Cigarro
from .func_smoothing import Func_Smoothing

__all__ = [
    "Function",
    "Func_Sphere",
    "Func_Rosen",
    "Func_Griew",
    "Func_Cigarro",
    "Func_Smoothing",
]
