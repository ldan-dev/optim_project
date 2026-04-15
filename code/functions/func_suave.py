"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/03/2026  (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: OPTIMIZACIÓN
DESCRIPCIÓN: Función de suavizado de imagen
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import diags
from function import Function 
from plot import Plot

class FunSuave(Function):
    """
    Clase hija que hereda de Function. 
    """
    def __init__(self, imag_ori: np.ndarray, lmbda: float = 0.1):
        super().__init__(params=None) 
    
        self.O = np.asarray(imag_ori, dtype=float) 
        self.lmbda = lmbda
        self.forma = self.O.shape
        self.num_pixeles = self.O.size
        self._hessiano = None 

    def eval(self, x: np.ndarray) -> float:
        """ Evalúa la función objetivo f(x) (Escalar) """
        x = np.asarray(x, dtype=float)
        if not isinstance(x, np.ndarray):
            raise TypeError("x debe ser un arreglo de numpy")
            
        X = x.reshape(self.forma)
        
        fidelidad = np.sum((self.O - X)**2)
        
        dif_horizontal = np.diff(X, axis=1)**2
        dif_vertical = np.diff(X, axis=0)**2
        regularizacion = np.sum(dif_horizontal) + np.sum(dif_vertical)
        
        return float(fidelidad + self.lmbda * regularizacion)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """ 
        Cálculo del Gradiente
        """
        x = np.asarray(x, dtype=float)
        X = x.reshape(self.forma)
        
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])
        
        laplaciano = convolve2d(X, kernel, mode='same', boundary='symm')
        grad = -2 * (self.O - X) + 4 * self.lmbda * laplaciano
        
        return grad.flatten()

    def ddiff(self) -> np.ndarray:
        """
        Cálculo de la Matriz Hessiana
        """
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

    def plot_2d(self, limites: list[float], lienzo: Plot):
        """ Gráfica de la función """
        pass

def main(): 
    """  
    Prueba para la clase FunSuave
    """ 
    import matplotlib.pyplot as plt
    
    try:
        imagen_original = plt.imread('lena.pgm')
        if imagen_original.max() <= 1.0:
            imagen_original = imagen_original * 255.0
    except FileNotFoundError:
        print("Aviso: No se encontró 'lena.pgm'")
        imagen_original = np.random.randint(0, 256, (128, 128))

    ruido = np.random.normal(0, 25, imagen_original.shape)
    imagen_ruidosa = np.clip(imagen_original + ruido, 0, 255)

    f = FunSuave(imag_ori=imagen_ruidosa, lmbda=0.5)

    print("Suavizando imagen...")
    x_actual = imagen_ruidosa.flatten()
    alpha = 0.05 
    
    for iteracion in range(50):
        grad = f.diff(x_actual)
        x_actual = x_actual - alpha * grad

    imagen_suavizada = x_actual.reshape(f.forma)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imagen_ruidosa, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Imagen Original (Con Ruido)')
    axes[0].axis('off')
    
    axes[1].imshow(imagen_suavizada, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Imagen Suavizada ($\lambda$={f.lmbda})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    main()
