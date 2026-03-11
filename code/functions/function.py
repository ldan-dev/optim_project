"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: OPTIMIZACIÓN
DESCRIPTION: 

"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import diags
from plot import Plot

class Function(): 
    def __init__(self, original_image: np.ndarray, lmbda: float = 0.1):
        """
        Inicializa la función con la imagen y el parámetro lambda.
        Args:
            (np.ndarray): lena.pgm como matriz 2D.
            (float): Parámetro lambda > 0.
        """
        self.O = np.asarray(original_image, dtype=float)
        self.lmbda = lmbda
        self.shape = self.O.shape
        self.n_pixels = self.O.size
        
        # Hessiano es constante para esta función
        self._hessian = None 

    def eval(self, x: np.ndarray) -> float:
        """ Evalúa la función objetivo f(x) """
        x = np.asarray(x, dtype=float)
        if not isinstance(x, np.ndarray):
            raise TypeError("x debe ser numpy.ndarray")
        
        X = x.reshape(self.shape)
        
        fidelidad = np.sum((self.O - X)**2)
        
        # Se calcula la diferencia entre píxeles adyacentes
        diff_horizontal = np.diff(X, axis=1)**2
        diff_vertical = np.diff(X, axis=0)**2
        regularizacion = np.sum(diff_horizontal) + np.sum(diff_vertical)
        
        # f(x) = Fidelidad + lambda * Regularización
        return float(fidelidad + self.lmbda * regularizacion)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """ 1ra derivada (Gradiente) """
        X = x.reshape(self.shape)
        
        # Operador Laplaciano discreto para calcular las diferencias de los vecinos
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])
        laplaciano = convolve2d(X, kernel, mode='same', boundary='symm')
        
        # Gradiente: -2(O_ij - X_ij) + 4*lambda*(Laplaciano)
        grad = -2 * (self.O - X) + 4 * self.lmbda * laplaciano
        
        # Gradiente como vector 1D para el optimizador
        return grad.flatten()

    def ddiff(self) -> np.ndarray:
        """ 2da derivada (Hessiano) """
        if self._hessian is not None:
            return self._hessian
            
        cols = self.shape[1]
        
        # Diagonal principal: 2 + 16*lambda
        main_diag = np.full(self.n_pixels, 2 + 16 * self.lmbda)
        
        # Diagonales secundarias: -4*lambda
        off_diag_1 = np.full(self.n_pixels - 1, -4 * self.lmbda)
        off_diag_cols = np.full(self.n_pixels - cols, -4 * self.lmbda)
        
        # Ajuste matemático: Evitar que el último píxel de una fila 
        # se conecte con el primer píxel de la siguiente fila
        off_diag_1[cols-1::cols] = 0 
        
        H = diags(
            diagonals=[main_diag, off_diag_1, off_diag_1, off_diag_cols, off_diag_cols],
            offsets=[0, 1, -1, cols, -cols],
            format="csr"
        )
        
        self._hessian = H
        return self._hessian

    def plot_2d(self, lim: list[float], canva: Plot):
        """ Plot de la función """
        
        pass
