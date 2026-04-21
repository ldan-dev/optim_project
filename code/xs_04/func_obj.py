"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: Optimización
DESCRIPTION:
    Función objetivo SSD para registro afín de imágenes.

Formulación exacta del ejercicio:
    f(t) = 1/2 * sum_{i=1}^n sum_{j=1}^m  (v_o_{i,j} - v_m_{i,j}*)^2

    X(p_{i,j}, t) = [i, j, 1, 0, 0, 0 ] * [t1]
                    [0, 0, 0, i, j, 1 ]   [t2]
                                           [t3]
                                           [t4]
                                           [t5]
                                           [t6]
    =>  x* = t1*i + t2*j + t3    (i = fila, j = columna)
        y* = t4*i + t5*j + t6

    Gradiente (regla de la cadena):
        df/dt_k = -sum_{i,j} (v_o_{i,j} - v_m_{i,j}*) * dv_m*/dt_k

        dv_m*/dt_k = (∂Im/∂x*) * (∂x*/∂t_k)  +  (∂Im/∂y*) * (∂y*/∂t_k)

    D_t v_m* = X^T * [∂Im/∂x*, ∂Im/∂y*]^T   =>  vector 6x1:
        [i * gx*, j * gx*, gx*, i * gy*, j * gy*, gy*]

    donde gx* = ∂Im/∂x* evaluado en (x*, y*) = gradiente de Im en dirección
    de filas (eje-0), y gy* en dirección de columnas (eje-1).

    CONVENCIÓN: i = fila (eje-0), j = columna (eje-1)
                x* -> eje fila,   y* -> eje columna
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
    """

    def __init__(self, fixed_img: np.ndarray, moving_img: np.ndarray):
        super().__init__()

        self.fixed_img  = np.asarray(fixed_img,  dtype=float)
        self.moving_img = np.asarray(moving_img, dtype=float)

        if self.fixed_img.shape != self.moving_img.shape:
            raise ValueError("Las imágenes fixed y moving deben tener la misma forma.")

        # Gradiente de Im: np.gradient(f) => [df/d_eje0, df/d_eje1]
        #   eje-0 = filas (dirección i)   => gx (∂Im/∂x*, donde x* es fila)
        #   eje-1 = columnas (dirección j)=> gy (∂Im/∂y*, donde y* es columna)
        self.gx_img, self.gy_img = np.gradient(self.moving_img)

        # Malla i (filas) y j (columnas) de la imagen fija
        h, w = self.fixed_img.shape
        # i_coords[i,j] = i,  j_coords[i,j] = j
        self.i_coords, self.j_coords = np.meshgrid(
            np.arange(h), np.arange(w), indexing='ij'
        )
        self.i_coords = self.i_coords.astype(float)
        self.j_coords = self.j_coords.astype(float)

        self.name = "IntensityBasedAffineRegistration"

    # ------------------------------------------------------------------
    def _warp_and_gradients(self, theta: np.ndarray):
        """
        Warp: para cada pixel (i,j) de la imagen fija calcula su posición
        transformada (x*, y*) en la imagen móvil y muestrea Im y su gradiente.

        x* = t1*i + t2*j + t3   (coordenada de fila en Im)
        y* = t4*i + t5*j + t6   (coordenada de columna en Im)

        map_coordinates recibe [eje0, eje1] = [x*, y*]
        """
        theta = AffineModel6.validate_theta(theta)
        t1, t2, t3, t4, t5, t6 = theta

        x_warp = t1 * self.i_coords + t2 * self.j_coords + t3  # fila en Im
        y_warp = t4 * self.i_coords + t5 * self.j_coords + t6  # col  en Im

        coords = np.stack([x_warp, y_warp])  # shape (2, H, W)

        warped_img = map_coordinates(self.moving_img, coords, order=1, mode='constant', cval=0.0)
        warped_gx  = map_coordinates(self.gx_img,    coords, order=1, mode='constant', cval=0.0)
        warped_gy  = map_coordinates(self.gy_img,    coords, order=1, mode='constant', cval=0.0)

        return warped_img, warped_gx, warped_gy

    # ------------------------------------------------------------------
    def eval(self, theta: np.ndarray) -> float:
        """f(t) = 1/2 * sum_{i,j} (v_o - v_m*)^2"""
        warped_img, _, _ = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img
        return 0.5 * float(np.sum(residual ** 2))

    # ------------------------------------------------------------------
    def diff(self, theta: np.ndarray) -> np.ndarray:
        """
        Gradiente analítico:
            df/dt_k = -sum_{i,j} residual_{i,j} * (D_t v_m*)_k

        D_t v_m* = X^T * [gx*, gy*]^T  con X = [i,j,1,0,0,0; 0,0,0,i,j,1]

        Componentes:
            dt1: gx* * i
            dt2: gx* * j
            dt3: gx* * 1
            dt4: gy* * i
            dt5: gy* * j
            dt6: gy* * 1
        """
        warped_img, warped_gx, warped_gy = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img  # (H, W)

        i = self.i_coords  # fila
        j = self.j_coords  # columna

        grad = np.zeros(AffineModel6.N_PARAMS, dtype=float)
        grad[0] = -np.sum(residual * warped_gx * i)
        grad[1] = -np.sum(residual * warped_gx * j)
        grad[2] = -np.sum(residual * warped_gx)
        grad[3] = -np.sum(residual * warped_gy * i)
        grad[4] = -np.sum(residual * warped_gy * j)
        grad[5] = -np.sum(residual * warped_gy)

        return grad

    def ddiff(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BFGS aproxima el Hessiano.")

    @classmethod
    def from_images_placeholder(cls, moving_img, fixed_img, step=12):
        return cls(fixed_img=fixed_img, moving_img=moving_img)


def main():
    """Prueba: verifica que f=0 y |grad|=0 cuando fixed==moving."""
    from scipy.optimize import check_grad
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DIR = os.path.dirname(os.path.abspath(__file__))

    def load_gray(p):
        img = plt.imread(p).astype(float)
        if img.ndim == 3: img = img[..., 0]
        vmax = img.max(); return img / vmax if vmax > 0 else img

    fixed  = load_gray(os.path.join(DIR, "I_1.pgm"))
    moving = load_gray(os.path.join(DIR, "I_6.pgm"))

    obj = AffineRegistrationObjective(fixed, moving)
    theta0 = AffineModel6.identity_theta()

    err = check_grad(obj.eval, obj.diff, theta0)
    print(f"Error gradiente (dif. finitas): {err:.4e}")
    print(f"f(theta_id) = {obj.eval(theta0):.4f}")
    print(f"|grad|      = {np.linalg.norm(obj.diff(theta0)):.4f}")

    # Caso trivial: fixed == moving => f=0, grad=0
    obj2 = AffineRegistrationObjective(fixed, fixed)
    print(f"\nfixed==moving => f={obj2.eval(theta0):.6f}, |grad|={np.linalg.norm(obj2.diff(theta0)):.6f}")


if __name__ == "__main__":
    main()
