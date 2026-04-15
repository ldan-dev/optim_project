r"""
Función de suavizado para imágenes PGM.

Modelo cuadrático:

f(x) = \sum_{i,j} (x_ij - o_ij)^2 + 2*lambda*\sum_{(i,j)~(k,l)} (x_ij - x_kl)^2

Donde (i,j)~(k,l) son vecinos 4-conectados.
Este modelo produce:
    grad = Hx - b
con Hessiana SPD constante.
"""

import numpy as np

try:
    from .function import Function
except ImportError:
    from function import Function


class Func_Smoothing(Function):
    def __init__(self, original_img: np.ndarray, lam: float = 0.2):
        super().__init__()

        if lam <= 0:
            raise ValueError("lam debe ser > 0")

        original_img = np.asarray(original_img, dtype=float)
        if original_img.ndim != 2:
            raise ValueError("original_img debe ser una matriz 2D")

        self.original_img = original_img
        self.lam = lam
        self.h, self.w = original_img.shape
        self.n = self.h * self.w

        self.b = 2.0 * self.original_img.reshape(-1)
        self.diag = self._build_hessian_diag()
        self.name = "ImageSmoothing"

    def _build_hessian_diag(self) -> np.ndarray:
        diag = np.full(self.n, 2.0, dtype=float)

        for i in range(self.h):
            for j in range(self.w):
                idx = i * self.w + j
                neighbors = 0
                if i > 0:
                    neighbors += 1
                if i < self.h - 1:
                    neighbors += 1
                if j > 0:
                    neighbors += 1
                if j < self.w - 1:
                    neighbors += 1

                diag[idx] += 4.0 * self.lam * neighbors

        return diag

    def _neighbors(self, i: int, j: int):
        if i > 0:
            yield i - 1, j
        if i < self.h - 1:
            yield i + 1, j
        if j > 0:
            yield i, j - 1
        if j < self.w - 1:
            yield i, j + 1

    def apply_hessian(self, x: np.ndarray) -> np.ndarray:
        """Calcula Hx sin formar H densa (matrix-free)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"x debe tener tamaño {self.n}")

        x_img = x.reshape(self.h, self.w)
        hx = np.zeros_like(x_img, dtype=float)

        for i in range(self.h):
            for j in range(self.w):
                center = x_img[i, j]
                value = 2.0 * center
                for ni, nj in self._neighbors(i, j):
                    value += 4.0 * self.lam * (center - x_img[ni, nj])
                hx[i, j] = value

        return hx.reshape(-1)

    def eval(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"x debe tener tamaño {self.n}")

        x_img = x.reshape(self.h, self.w)
        fidelity = np.sum((x_img - self.original_img) ** 2)

        smooth = 0.0
        smooth += np.sum((x_img[:, 1:] - x_img[:, :-1]) ** 2)
        smooth += np.sum((x_img[1:, :] - x_img[:-1, :]) ** 2)

        return fidelity + 2.0 * self.lam * smooth

    def diff(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"x debe tener tamaño {self.n}")
        return self.apply_hessian(x) - self.b

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """
        Retorna Hessiana densa (útil para pruebas pequeñas).
        Para imágenes grandes, usa apply_hessian().
        """
        _ = np.asarray(x, dtype=float).reshape(-1)
        H = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.h):
            for j in range(self.w):
                idx = i * self.w + j
                neighbors = list(self._neighbors(i, j))
                H[idx, idx] = 2.0 + 4.0 * self.lam * len(neighbors)
                for ni, nj in neighbors:
                    nidx = ni * self.w + nj
                    H[idx, nidx] = -4.0 * self.lam

        return H

    def flatten_image(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=float).reshape(-1)

    def unflatten_image(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"x debe tener tamaño {self.n}")
        return x.reshape(self.h, self.w)
