"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION:
Función objetivo base para registro afín por correspondencias.
"""

import os
import sys
import numpy as np

# Permite ejecutar `python code/xs_04/xs04_main.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from functions import Function
from xs_04.modelo import AffineModel6


class AffineRegistrationObjective(Function):
    """
    f(theta) = 1/2 ||X theta - y||^2

    - X: matriz de diseño armada con puntos de la imagen móvil.
    - y: puntos objetivo en imagen fija.
    """

    def __init__(self, moving_points_xy: np.ndarray, fixed_points_xy: np.ndarray):
        super().__init__()

        moving_points_xy = np.asarray(moving_points_xy, dtype=float)
        fixed_points_xy = np.asarray(fixed_points_xy, dtype=float)

        if moving_points_xy.shape != fixed_points_xy.shape:
            raise ValueError("moving_points_xy y fixed_points_xy deben tener misma forma")
        if moving_points_xy.ndim != 2 or moving_points_xy.shape[1] != 2:
            raise ValueError("Las correspondencias deben tener forma (N, 2)")

        self.moving_points = moving_points_xy
        self.fixed_points = fixed_points_xy
        self.X = AffineModel6.build_design_matrix(self.moving_points)
        self.y = AffineModel6.stack_targets(self.fixed_points)
        self.name = "AffineRegistrationLS"

    def eval(self, theta: np.ndarray) -> float:
        theta = AffineModel6.validate_theta(theta)
        residual = self.X @ theta - self.y
        return 0.5 * float(residual @ residual)

    def diff(self, theta: np.ndarray) -> np.ndarray:
        theta = AffineModel6.validate_theta(theta)
        residual = self.X @ theta - self.y
        return self.X.T @ residual

    def ddiff(self, theta: np.ndarray) -> np.ndarray:
        _ = AffineModel6.validate_theta(theta)
        return self.X.T @ self.X

    @staticmethod
    def build_grid_correspondences(shape: tuple[int, int], step: int = 12):
        """
        Base para generar puntos en malla regular.
        NOTA: aquí aún no hacemos matching entre imágenes.
        """
        h, w = shape
        ys = np.arange(0, h, step, dtype=float)
        xs = np.arange(0, w, step, dtype=float)
        xv, yv = np.meshgrid(xs, ys)
        grid = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
        return grid

    @classmethod
    def from_images_placeholder(
        cls,
        moving_img: np.ndarray,
        fixed_img: np.ndarray,
        step: int = 12,
    ):
        """
        Constructor base para pipeline con imágenes.
        TODO: reemplazar por extracción de correspondencias reales (SIFT/ORB/flujo óptico).
        """
        if moving_img.shape != fixed_img.shape:
            raise ValueError("Para esta base, moving_img y fixed_img deben tener misma forma")

        grid = cls.build_grid_correspondences(moving_img.shape[:2], step=step)

        # Base mínima: identidad como correspondencia inicial.
        # Luego aquí conectas detector + matcher y sustituyes fixed_pts.
        moving_pts = grid.copy()
        fixed_pts = grid.copy()
        return cls(moving_points_xy=moving_pts, fixed_points_xy=fixed_pts)

def main():
    """Prueba mínima de la función objetivo."""
    moving = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    fixed = np.array([[2, -1], [3, -1], [2, 0], [3, 0]], dtype=float)
    obj = AffineRegistrationObjective(moving, fixed)
    theta0 = np.array([1, 0, 0, 1, 0, 0], dtype=float)
    print("f(theta0)=", obj.eval(theta0))


if __name__ == "__main__":
    main()