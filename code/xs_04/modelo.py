"""
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA:
DESCRIPTION: Modelo de transformación afín 2D con 6 parámetros para registro de imágenes.
"""

import numpy as np

_THETA_SIZE = 6


class AffineModel6:
    """
    Modelo afín 2D con 6 parámetros::

        [x*]   [t1  t2] [x]   [t5]
        [y*] = [t3  t4] [y] + [t6]

    theta = [t1, t2, t3, t4, t5, t6]

    Convención de índices
    ----------------------
    0: a11   1: a12   (primera fila de A)
    2: a21   3: a22   (segunda fila de A)
    4: tx    5: ty    (traslación)
    """

    N_PARAMS: int = _THETA_SIZE 

    # ------------------------------------------------------------------ #
    #  Construcción / validación                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def identity_theta() -> np.ndarray:  
        """Devuelve el theta de transformación identidad (sin cambio)."""
        return np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    @staticmethod
    def validate_theta(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != _THETA_SIZE:
            raise ValueError(f"theta debe tener {_THETA_SIZE} parámetros, recibió {theta.size}")
        return theta

    @staticmethod
    def theta_to_matrix(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Descompone theta en la matriz lineal A (2×2) y el vector de traslación b (2,)."""
        theta = AffineModel6.validate_theta(theta)
        A = np.array([[theta[0], theta[1]],
                      [theta[2], theta[3]]])
        b = np.array([theta[4], theta[5]])
        return A, b

    # ------------------------------------------------------------------ #
    #  Transformación                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def transform_points(points_xy: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Aplica la transformación afín a una nube de puntos.

        Parameters
        ----------
        points_xy : ndarray, shape (N, 2)
            Puntos en coordenadas (x, y).
        theta : array-like, shape (6,)
            Parámetros del modelo.

        Returns
        -------
        ndarray, shape (N, 2)
            Puntos transformados.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")

        A, b = AffineModel6.theta_to_matrix(theta)
        return points_xy @ A.T + b  # equivalente a (A @ point.T).T + b

    # ------------------------------------------------------------------ #
    #  Sistema lineal por correspondencias                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_design_matrix(points_xy: np.ndarray) -> np.ndarray:
        """
        Construye la matriz de diseño D (2N × 6) para resolver theta por mínimos cuadrados
        dado un conjunto de correspondencias puntuales.

        Cada par de filas codifica un punto (x, y)::

            fila 2i   → [x  y  0  0  1  0]   (ecuación para x*)
            fila 2i+1 → [0  0  x  y  0  1]   (ecuación para y*)

        Uso:  D @ theta = stack_targets(puntos_destino)
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")

        n = points_xy.shape[0]
        D = np.zeros((2 * n, _THETA_SIZE), dtype=float)

        x = points_xy[:, 0]
        y = points_xy[:, 1]

        D[0::2, 0] = x   # t1 · x
        D[0::2, 1] = y   # t2 · y
        D[0::2, 4] = 1.0 # t5 (tx)

        D[1::2, 2] = x   # t3 · x
        D[1::2, 3] = y   # t4 · y
        D[1::2, 5] = 1.0 # t6 (ty)

        return D

    @staticmethod
    def stack_targets(points_xy: np.ndarray) -> np.ndarray:
        """
        Convierte puntos (N, 2) al vector objetivo [x1, y1, x2, y2, ...] (2N,)
        para usarse como lado derecho en el sistema D @ theta = b.
        """
        points_xy = np.asarray(points_xy, dtype=float)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy debe tener forma (N, 2)")
        return points_xy.reshape(-1)

    @staticmethod
    def solve_from_correspondences(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> np.ndarray:  #resuelve theta directo desde correspondencias
        """
        Estima theta por mínimos cuadrados dados pares de puntos correspondientes.

        Parameters
        ----------
        src : ndarray, shape (N, 2)  — puntos en la imagen móvil
        dst : ndarray, shape (N, 2)  — puntos correspondientes en la imagen fija

        Returns
        -------
        theta : ndarray, shape (6,)

        Notes
        -----
        Necesita N >= 3 pares de puntos no colineales para solución única.
        """
        D = AffineModel6.build_design_matrix(src)
        b = AffineModel6.stack_targets(dst)
        theta, *_ = np.linalg.lstsq(D, b, rcond=None)
        return theta


# --------------------------------------------------------------------------- #
#  Prueba                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=== identity_theta ===")
    theta_id = AffineModel6.identity_theta()
    print("theta identidad:", theta_id)

    pts = np.array([[0.0, 0.0], [10.0, 5.0], [3.0, 8.0]])
    print("\n=== transform_points (identidad) ===")
    print(AffineModel6.transform_points(pts, theta_id))  # debe devolver pts sin cambio

    theta = np.array([1.0, 0.1, -0.1, 1.0, 2.0, -3.0])
    print("\n=== transform_points (theta con rotación + traslación) ===")
    transformed = AffineModel6.transform_points(pts, theta)
    print(transformed)

    print("\n=== solve_from_correspondences (round-trip) ===")
    # Si damos como destino los puntos ya transformados, debe recuperar theta exactamente
    theta_recovered = AffineModel6.solve_from_correspondences(pts, transformed)
    print("theta original: ", theta)
    print("theta recuperado:", np.round(theta_recovered, 10))
    print("¿Coinciden?", np.allclose(theta, theta_recovered))


if __name__ == "__main__":
    main()