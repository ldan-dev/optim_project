"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION:
Función objetivo para registro afín de imágenes basado en INTENSIDAD (SSD).
"""

import os
import sys
import numpy as np
from scipy.ndimage import map_coordinates

# Permite ejecutar `python code/xs_04/xs04_main.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from functions import Function
from xs_04.modelo import AffineModel6


class AffineRegistrationObjective(Function):
    """
    f(t) = 1/2 * || I0 - X(Im, t) ||^2
    
    Implementación basada en la Suma de Diferencias al Cuadrado (SSD)
    evaluando la intensidad de todos los píxeles de la imagen.
    """

    def __init__(self, fixed_img: np.ndarray, moving_img: np.ndarray):
        super().__init__()
        
        self.fixed_img = np.asarray(fixed_img, dtype=float)
        self.moving_img = np.asarray(moving_img, dtype=float)

        if self.fixed_img.shape != self.moving_img.shape:
            raise ValueError("Las imágenes fixed y moving deben tener la misma forma.")

        # 1. Gradiente espacial (D_t V_m*) mediante diferencias centrales
        # np.gradient aproxima exactamente la ecuación: (v[i+1] - v[i-1]) / 2
        self.grad_y, self.grad_x = np.gradient(self.moving_img)

        # 2. Malla de coordenadas de la imagen fija (I0)
        h, w = self.fixed_img.shape
        self.y_coords, self.x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        self.name = "IntensityBasedAffineRegistration"

    def _warp_and_gradients(self, theta: np.ndarray):
        """
        Aplica la transformación afín a las coordenadas de la malla y 
        muestrea la imagen móvil y sus gradientes interpolando los valores.
        """
        theta = AffineModel6.validate_theta(theta)
        
        # Convención estricta de AffineModel6:
        t1, t2, t3, t4, t5, t6 = theta

        # x* = t1*x + t2*y + t5
        # y* = t3*x + t4*y + t6
        X_warp = self.x_coords * t1 + self.y_coords * t2 + t5
        Y_warp = self.x_coords * t3 + self.y_coords * t4 + t6

        # map_coordinates [filas, columnas] -> [Y, X]
        coords = np.stack([Y_warp, X_warp])

        warped_img = map_coordinates(self.moving_img, coords, order=1, mode='constant', cval=0.0)
        warped_gx = map_coordinates(self.grad_x, coords, order=1, mode='constant', cval=0.0)
        warped_gy = map_coordinates(self.grad_y, coords, order=1, mode='constant', cval=0.0)

        return warped_img, warped_gx, warped_gy

    def eval(self, theta: np.ndarray) -> float:
        warped_img, _, _ = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img
        
        # f(t) = 1/2 sum( (V_o - V_m*)^2 )
        return 0.5 * float(np.sum(residual**2))

    def diff(self, theta: np.ndarray) -> np.ndarray:
        warped_img, warped_gx, warped_gy = self._warp_and_gradients(theta)
        residual = self.fixed_img - warped_img

        x = self.x_coords
        y = self.y_coords

        # Calculamos d V_m* / dt = X^T * [grad_x, grad_y]^T
        # Respetando los índices [t1, t2, t3, t4, t5, t6] de AffineModel6:
        dV_dt0 = warped_gx * x
        dV_dt1 = warped_gx * y
        dV_dt2 = warped_gy * x
        dV_dt3 = warped_gy * y
        dV_dt4 = warped_gx * 1.0  # Traslación X (t5)
        dV_dt5 = warped_gy * 1.0  # Traslación Y (t6)

        # Gradiente: df(t)/dt = - sum( residual * (d V_m* / dt) )
        # Desaparece el multiplicador 2 por la cancelación con el 1/2 de f(t)
        grad = np.zeros(AffineModel6.N_PARAMS, dtype=float)
        grad[0] = -np.sum(residual * dV_dt0)
        grad[1] = -np.sum(residual * dV_dt1)
        grad[2] = -np.sum(residual * dV_dt2)
        grad[3] = -np.sum(residual * dV_dt3)
        grad[4] = -np.sum(residual * dV_dt4)
        grad[5] = -np.sum(residual * dV_dt5)

        return grad

    def ddiff(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError("El algoritmo BFGS aproxima el Hessiano no se requiere")

    @classmethod
    def from_images_placeholder(cls, moving_img: np.ndarray, fixed_img: np.ndarray, step: int = 12):
        return cls(fixed_img=fixed_img, moving_img=moving_img)


def main():
    """Prueba mínima de la función objetivo con matrices de prueba."""
    fixed = np.ones((10, 10))
    moving = np.ones((10, 10)) * 0.5
    
    obj = AffineRegistrationObjective(fixed, moving)
    theta0 = AffineModel6.identity_theta()
    
    print("=== Evaluación Inicial ===")
    print(f"f(theta0) = {obj.eval(theta0)}")
    print(f"Gradiente(theta0) = {obj.diff(theta0)}")

if __name__ == "__main__":
    main()
