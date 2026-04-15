"""
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""

import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plot_image import cargar_imagen
from xs_04.func_obj import AffineRegistrationObjective
from xs_04.bfgs import BFGSOptimizer
from xs_04.plot_registration import RegistrationPlot


def _to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[..., 0]
    raise ValueError("Formato de imagen no soportado")


def main():
    parser = argparse.ArgumentParser(description="Base de registro afín con BFGS")
    parser.add_argument("--fixed", type=str, required=True, help="Ruta imagen fija")
    parser.add_argument("--moving", type=str, required=True, help="Ruta imagen móvil")
    parser.add_argument("--max-it", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--step", type=int, default=16, help="Paso de malla para correspondencias base")
    args = parser.parse_args()

    fixed_img = _to_gray(cargar_imagen(args.fixed))
    moving_img = _to_gray(cargar_imagen(args.moving))

    # Base inicial: correspondencias por rejilla (placeholder).
    func = AffineRegistrationObjective.from_images_placeholder(
        moving_img=moving_img,
        fixed_img=fixed_img,
        step=args.step,
    )

    solver = BFGSOptimizer(
        func=func,
        max_it=args.max_it,
        tolerance=args.tol,
        cond_step="wolfe_strong",
        descent_dir="gradient",
    )

    theta0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    theta_opt = solver.solve(theta0)

    print("Theta encontrado (base):")
    for i, value in enumerate(theta_opt, start=1):
        print(f"theta{i}: {value:.6f}")

    # Nota: aquí aún no aplicamos warp real de imagen (solo base de estructura).
    plotter = RegistrationPlot(title="Registro afín (estructura base)")
    plotter.show(fixed_img=fixed_img, moving_img=moving_img, theta=theta_opt, warped_img=None)


if __name__ == "__main__":
    main()