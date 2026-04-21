"""
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""

import numpy as np


class AffineModel6:
    """
    Modelo afín 2D con 6 parámetros según la formulación del ejercicio:

        X(p_i, t) = [x, y, 1, 0, 0, 0]   [t1]
                    [0, 0, 0, x, y, 1] * [t2]
                                          [t3]
                                          [t4]
                                          [t5]
                                          [t6]

    Esto produce:
        x* = t1*x + t2*y + t3
        y* = t4*x + t5*y + t6

    theta = [t1, t2, t3, t4, t5, t6]
    """

    N_PARAMS = 6

    @staticmethod
    def validate_theta(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != 6:
            raise ValueError("theta debe tener 6 parámetros")
        return theta

    @classmethod
    def identity_theta(cls) -> np.ndarray:
        """Devuelve theta identidad: x*=x, y*=y sin traslación."""
        # t1=1, t2=0, t3=0 (x* = 1*x + 0*y + 0)
        # t4=0, t5=1, t6=0 (y* = 0*x + 1*y + 0)
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)

    @staticmethod
    def theta_to_matrix(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convierte theta a (A, b) donde x*_vec = A @ x_vec + b.
            A = [[t1, t2],   b = [t3]
                 [t4, t5]]        [t6]
        """
        theta = AffineModel6.validate_theta(theta)
        A = np.array([
            [theta[0], theta[1]],
            [theta[3], theta[4]],
        ])
        b = np.array([theta[2], theta[5]])
        return A, b

    @staticmethod
    def transform_points(points_xy: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Aplica transformación afín a una nube de puntos (N,2)."""
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")

        A, b = AffineModel6.theta_to_matrix(theta)
        return points_xy @ A.T + b

    @staticmethod
    def build_design_matrix(points_xy: np.ndarray) -> np.ndarray:
        """
        Construye la matriz de diseño D_t X para la malla de píxeles.

        Para cada punto (x, y):
            fila par  -> [x, y, 1, 0, 0, 0]   (componente x*)
            fila impar -> [0, 0, 0, x, y, 1]  (componente y*)
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")

        n = points_xy.shape[0]
        X = np.zeros((2 * n, 6), dtype=float)

        x = points_xy[:, 0]
        y = points_xy[:, 1]

        # Fila par: componente x* = t1*x + t2*y + t3
        X[0::2, 0] = x
        X[0::2, 1] = y
        X[0::2, 2] = 1.0

        # Fila impar: componente y* = t4*x + t5*y + t6
        X[1::2, 3] = x
        X[1::2, 4] = y
        X[1::2, 5] = 1.0

        return X

    @staticmethod
    def stack_targets(points_xy: np.ndarray) -> np.ndarray:
        """Convierte puntos (N,2) a vector [x1,y1,x2,y2,...]."""
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")
        return points_xy.reshape(-1)


def main():
    """Prueba rápida de estructura base del modelo."""
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [3.0, 8.0]])
    # Identidad: x*=x, y*=y
    theta_id = AffineModel6.identity_theta()
    print("Transform con identidad (debe ser igual a pts):",
          AffineModel6.transform_points(pts, theta_id))

    theta = np.array([1.0, 0.1, 0.0, -0.1, 1.0, 0.0])
    print("Transform con rotación leve:", AffineModel6.transform_points(pts, theta))


if __name__ == "__main__":
    main()