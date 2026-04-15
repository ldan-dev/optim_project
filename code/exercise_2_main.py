"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 25/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
main del ejercicio 2

TODO: alguno cree una rama que se llame xs02/main
"""

import matplotlib.pyplot as plt
import numpy as np

from plot import Plot # clase Plot
from functions import *
from gradient_descend import GradientDescent

# # condiciones de paso y direcciones de descenso
# from step_conditions import CONDITIONS as STEP_CONDITIONS
# from descent_dir import DIRECTIONS as DESCENT_DIRECTIONS

def main():
    """  Entregable del ejercicio 2 """
    
    griew = Func_Griew()
    
    # 2. Configurar Newton con Griewangk
    newton_solver = GradientDescent(
        func=griew,
        step_size=1.0,
        max_it=100,
        tolerance=1e-6,
        cond_step="armijo",
        descent_dir="newton"  # Usar dirección de Newton
    )
    
    # 3. Resolver desde un punto inicial
    start_point = [0.5, 0.5]
    x_opt = newton_solver.solve(start_point)
    
    print(f"Punto inicial: {start_point}")
    print(f"Punto óptimo encontrado: {x_opt}")
    print(f"f(x*) = {griew.eval(x_opt)}")
    print(f"Iteraciones: {len(newton_solver.path) - 1}")
    
    # 4. Graficar curvas de nivel y trayectoria
    canvas = Plot(title="Método de Newton - Función Griewangk - Leonardo Daniel Aviña Neri")
    canvas.canvas(xlabel='x₁', ylabel='x₂')
    canvas.draw_contours(griew, range_val=griew.limits, density=200)
    canvas.draw_trace(newton_solver.path)
    canvas.show()


if __name__ == "__main__":
    main()
    