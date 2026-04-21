"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: Optimización
DESCRIPTION:
Función objetivo para registro afín de imágenes basado en INTENSIDAD (SSD).

Formulación:
    f(t) = 1/2 * sum_{i,j} (v_o_{i,j} - v_m_{i,j}*)^2

    X(p_i, t) = [x, y, 1, 0, 0, 0]   * [t1, t2, t3, t4, t5, t6]^T
                [0, 0, 0, x, y, 1]
    => x* = t1*x + t2*y + t3
       y* = t4*x + t5*y + t6

    D_t X(p_{i,j}) = [i, j, 1, 0, 0, 0]   (shape 6x2, transpuesta)
                     [0, 0, 0, i, j, 1]

    Gradiente (regla de la cadena):
        df/dt_k = -sum_{i,j} (v_o_{i,j} - v_m_{i,j}*) * (dv_m*/dt_k)
        dv_m*/dt_k = grad_Im(x*,y*)^T * (d X/d t_k)

        Vectorialmente:
        grad f(t) = -sum_{i,j} residual_{i,j} * D_t X^T * [dIm/dx*, dIm/dy*]^T
                  = -sum_{i,j} residual_{i,j} * (X^T * nabla_Im)_{i,j}
"""

import os
import sys
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from functions import Function
from xs_04.modelo import AffineModel6


class AffineRegistrationObjective(Function):
    """
    f(t) = 1/2 * || I_o - X(I_m, t) ||^2_2
         = 1/2 * sum_{i,j} (v_o_{i,j} - v_m_{i,j}*)^2

    donde v_m_{i,j}* = Im interpolada en la posición X(p_{i,j}, t).

    Parámetros (theta):
        [t1, t2, t3, t4, t5, t6]
        x* = t1*x + t2*y + t3
        y* = t4*x + t5*y + t6
    """

    def __init__(self, fixed_img: np.ndarray, moving_img: np.ndarray):
        super().__init__()

        self.fixed_img  = np.asarray(fixed_img,  dtype=float)
        self.moving_img = np.asarray(moving_img, dtype=float)

        if self.fixed_img.shape != self.moving_img.shape:
            raise ValueError("Las imágenes fixed y moving deben tener la misma forma.")

        # Gradiente de la imagen móvil (para la regla de la cadena).
        # np.gradient devuelve (grad_filas, grad_columnas) = (grad_y, grad_x)
        self.grad_y, self.grad_x = np.gradient(self.moving_img)

        # Malla de coordenadas (i = fila, j = columna) de la imagen fija.
        # Usamos indexing='ij' para que row_coords[i,j]=i, col_coords[i,j]=j
        h, w = self.fixed_img.shape
        self.row_coords, self.col_coords = np.meshgrid(
            np.arange(h), np.arange(w), indexing='ij'
        )
        # En la notación del ejercicio: x <-> col (j), y <-> row (i)
        self.x_coords = self.col_coords.astype(float)  # j
        self.y_coords = self.row_coords.astype(float)  # i

        self.name = "IntensityBasedAffineRegistration"

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _warp_and_gradients(self, theta: np.ndarray):
        """
        Aplica la transformación afín a cada píxel (x,y) de la malla y
        muestrea Im y sus gradientes (∂Im/∂x*, ∂Im/∂y*) por interpolación.

        X(p_{i,j}, t):
            x* = t1*x + t2*y + t3
            y* = t4*x + t5*y + t6
        (x=col, y=row)
        """
        theta = AffineModel6.validate_theta(theta)
        t1, t2, t3, t4, t5, t6 = theta

        # Posiciones transformadas
        X_warp = t1 * self.x_coords + t2 * self.y_coords + t3  # x* (col)
        Y_warp = t4 * self.x_coords + t5 * self.y_coords + t6  # y* (row)

        # map_coordinates espera [fila, columna] => [Y_warp, X_warp]
        coords = np.stack([Y_warp, X_warp])

        warped_img = map_coordinates(self.moving_img, coords, order=1, mode='constant', cval=0.0)
        warped_gx  = map_coordinates(self.grad_x,    coords, order=1, mode='constant', cval=0.0)
        warped_gy  = map_coordinates(self.grad_y,    coords, order=1, mode='constant', cval=0.0)

        return warped_img, warped_gx, warped_gy

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------
    def eval(self, theta: np.ndarray) -> float:
        """
        f(t) = 1/2 * sum_{i,j} (v_o_{i,j} - v_m_{i,j}*)^2
        """
        warped_img, _, _ = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img
        return 0.5 * float(np.sum(residual ** 2))

    # ------------------------------------------------------------------
    # Gradiente analítico
    # ------------------------------------------------------------------
    def diff(self, theta: np.ndarray) -> np.ndarray:
        """
        Gradiente de f respecto a theta.

        Por la regla de la cadena:
            df/dt_k = -sum_{i,j} residual_{i,j} * dv_m*/dt_k

        donde:
            dv_m*/dt_k = ∂Im/∂x* * ∂x*/∂t_k  +  ∂Im/∂y* * ∂y*/∂t_k

        La matriz jacobiana D_t X (6×2) para el píxel (x,y):
            D_t X^T = [x, y, 1, 0, 0, 0]   <- componente x*
                      [0, 0, 0, x, y, 1]   <- componente y*

        Entonces el vector de derivadas parciales dv_m*/dt (6×1):
            dv_m*/dt = D_t X^T * [∂Im/∂x*, ∂Im/∂y*]^T
                     = [gx*x, gx*y, gx*1, gy*x, gy*y, gy*1]

        Y el gradiente total:
            ∇f = -sum_{i,j} residual_{i,j} * [gx*x, gx*y, gx, gy*x, gy*y, gy]_{i,j}
        """
        warped_img, warped_gx, warped_gy = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img   # shape (H, W)

        x = self.x_coords   # col (j)
        y = self.y_coords   # row (i)

        # D_t X^T * [gx, gy]^T  ->  6 canales (H x W cada uno)
        # Componentes de x*: dt1, dt2, dt3
        dV_dt1 = warped_gx * x          # ∂Im/∂x* * ∂x*/∂t1 = gx * x
        dV_dt2 = warped_gx * y          # ∂Im/∂x* * ∂x*/∂t2 = gx * y
        dV_dt3 = warped_gx * 1.0        # ∂Im/∂x* * ∂x*/∂t3 = gx * 1
        # Componentes de y*: dt4, dt5, dt6
        dV_dt4 = warped_gy * x          # ∂Im/∂y* * ∂y*/∂t4 = gy * x
        dV_dt5 = warped_gy * y          # ∂Im/∂y* * ∂y*/∂t5 = gy * y
        dV_dt6 = warped_gy * 1.0        # ∂Im/∂y* * ∂y*/∂t6 = gy * 1

        # ∇f = -sum_{i,j} residual * dV/dt_k
        grad = np.zeros(AffineModel6.N_PARAMS, dtype=float)
        grad[0] = -np.sum(residual * dV_dt1)
        grad[1] = -np.sum(residual * dV_dt2)
        grad[2] = -np.sum(residual * dV_dt3)
        grad[3] = -np.sum(residual * dV_dt4)
        grad[4] = -np.sum(residual * dV_dt5)
        grad[5] = -np.sum(residual * dV_dt6)

        return grad

    def ddiff(self, theta: np.ndarray) -> np.ndarray:
        # BFGS aproxima el Hessiano, no es necesario calcularlo explícitamente.
        raise NotImplementedError("El algoritmo BFGS aproxima el Hessiano, no se requiere ddiff.")

    @classmethod
    def from_images_placeholder(cls, moving_img: np.ndarray, fixed_img: np.ndarray, step: int = 12):
        return cls(fixed_img=fixed_img, moving_img=moving_img)


def main():
    """Prueba mínima de la función objetivo con matrices de prueba."""
    rng = np.random.default_rng(42)
    fixed  = rng.uniform(0, 255, (20, 20))
    moving = rng.uniform(0, 255, (20, 20))

    obj = AffineRegistrationObjective(fixed, moving)

    # Theta identidad: no debería transformar nada
    theta0 = AffineModel6.identity_theta()

    print("=== Evaluación Inicial ===")
    print(f"f(theta_id) = {obj.eval(theta0):.4f}")
    print(f"Gradiente(theta_id) = {obj.diff(theta0)}")

    # Con la imagen fija == imagen móvil, f debe ser 0 y grad debe ser 0
    obj2 = AffineRegistrationObjective(fixed, fixed)
    print("\n=== fixed == moving (f debe ser 0) ===")
    print(f"f(theta_id) = {obj2.eval(theta0):.6f}")
    print(f"|grad| = {np.linalg.norm(obj2.diff(theta0)):.6f}")


if __name__ == "__main__":
    main()
