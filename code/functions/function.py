"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: OPTIMIZACIÓN
DESCRIPCIÓN: Clase Function para evaluar la función objetivo del suavizado de Lena.pgm,
             incluyendo el cálculo de su Gradiente y Hessiano.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import diags
from plot import Plot # clase Plot

class Function(): 
    def __init__(self, imag_ori: np.ndarray, lmbda: float = 0.1):
        """
        Inicializa la función con la imagen y el parámetro lambda.
        Argumentos:
            imag_ori (np.ndarray): lena.pgm como matriz 2D.
            lmbda (float): Parámetro lambda > 0.
        """
        # Transforma la imagen original en una matriz
        self.O = np.asarray(imag_ori, dtype=float) 
        self.lmbda = lmbda
        self.forma = self.O.shape
        self.num_pixeles = self.O.size # Guardamos las dimensiones de la imagen
        
        # Hessiano constante
        self._hessiano = None 

    def eval(self, x: np.ndarray) -> float:
        """ 
        Evalúa la función objetivo 
        f(x) (Escalar) 
        """
        x = np.asarray(x, dtype=float)
        if not isinstance(x, np.ndarray):
            raise TypeError("x debe ser un arreglo de numpy (numpy.ndarray)")
        
        X = x.reshape(self.forma)
        
        fidelidad = np.sum((self.O - X)**2)
        
        # Diferencia entre píxeles adyacentes
        dif_horizontal = np.diff(X, axis=1)**2
        dif_vertical = np.diff(X, axis=0)**2
        regularizacion = np.sum(dif_horizontal) + np.sum(dif_vertical)
        
        # f(x) = Fidelidad + lambda * Regularización
        return float(fidelidad + self.lmbda * regularizacion)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """ 
        Cálculo del Gradiente (1ra derivada) 
        """
        x = np.asarray(x, dtype=float)
        X = x.reshape(self.forma)
        
        # Operador Laplaciano discreto para calcular las diferencias de los vecinos
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])
        
        # Convolución para aplicar el kernel a toda la imagen rápidamente
        laplaciano = convolve2d(X, kernel, mode='same', boundary='symm')
        
        # Gradiente: -2(O_ij - X_ij) + 4*lambda*(Laplaciano)
        grad = -2 * (self.O - X) + 4 * self.lmbda * laplaciano
        
        # Retornamos el gradiente aplanado como vector 1D
        return grad.flatten()

    def ddiff(self) -> np.ndarray:
        """ 
        Cálculo de la Matriz Hessiana (2da derivada)
        """
        if self._hessiano is not None:
            return self._hessiano
            
        columnas = self.forma[1]
        
        # Diagonal principal: 2 + 16*lambda
        diag_principal = np.full(self.num_pixeles, 2 + 16 * self.lmbda)
        
        # Diagonales secundarias: -4*lambda
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

    def plot_2d(self, limites: list[float], lienzo: Plot):
        """ Gráfica de la función """
        pass

def main(): 
    """  Docstring for main  """ 

if __name__ == "__main__": 
    main()
