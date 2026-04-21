import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import diags

class Function_Smooth(): 
    def __init__(self, imag_ori: np.ndarray, lmbda: float = 0.1):
        self.O = np.asarray(imag_ori, dtype=float) 
        self.lmbda = lmbda
        self.forma = self.O.shape
        self.num_pixeles = self.O.size
        self._hessiano = None 

    def diff(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape(self.forma)
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])
        laplaciano = convolve2d(X, kernel, mode='same', boundary='symm')
        grad = 2 * (X - self.O) + 4 * self.lmbda * laplaciano
        return grad.flatten()

    def ddiff(self, x=None) -> np.ndarray: # <--- Cambio: acepta x
        if self._hessiano is not None:
            return self._hessiano
            
        columnas = self.forma[1]
        diag_principal = np.full(self.num_pixeles, 2 + 16 * self.lmbda)
        diag_vecinos = np.full(self.num_pixeles - 1, -4 * self.lmbda)
        diag_columnas = np.full(self.num_pixeles - columnas, -4 * self.lmbda)
        diag_vecinos[columnas-1::columnas] = 0 
        
        H = diags(
            diagonals=[diag_principal, diag_vecinos, diag_vecinos, diag_columnas, diag_columnas],
            offsets=[0, 1, -1, columnas, -columnas],
            format="csr"
        )
        self._hessiano = H
        return self._hessiano