"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: OPTIMIZACIÓN
DESCRIPCIÓN: Clase Function para evaluar la función objetivo del suavizado de Lena.pgm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plot import Plot # clase Plot

class Function(): 
    def __init__(self, imag_ori: np.ndarray, lmbda: float = 0.1):
        """
        Inicializa la función con la imagen y el parámetro lambda.
        Argumentos:
            imagen_original (np.ndarray): lena.pgm como matriz 2D.
            lmbda (float): Parámetro lambda > 0.
        """
        
        self.O = np.asarray(imag_ori, dtype=float) # Toma la imagen_original y la transforma en una matriz (de tipo flotantes)
        self.lmbda = lmbda
        self.forma = self.O.shape
        self.num_pixeles = self.O.size # Guardan las dimensiones de la imagen_original

    def eval(self, x: np.ndarray) -> float:
        """ 
        Evalúa la función objetivo 
        f(x) (Escalar) 
        """
        
        x = np.asarray(x, dtype=float)

        X = x.reshape(self.forma)
        
        fidelidad = np.sum((self.O - X)**2)
        
        # Se calcula la diferencia entre píxeles adyacentes
        dif_horizontal = np.diff(X, axis=1)**2
        dif_vertical = np.diff(X, axis=0)**2
        regularizacion = np.sum(dif_horizontal) + np.sum(dif_vertical)
        
        # f(x) = Fidelidad + lambda * Regularización
        return float(fidelidad + self.lmbda * regularizacion)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """ Gradiente """
        pass

    def ddiff(self) -> np.ndarray:
        """ Hessiano """
        pass

    def plot_2d(self, limites: list[float], lienzo: Plot):
        """ Gráfica de la función """
        pass
        
def main():
    """  Docstring for main  """

if __name__ == "__main__":
    main()
