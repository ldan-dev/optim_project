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
    """  
    Bloque de prueba para verificar que la función opere correctamente 
    y mostrar la imagen original vs suavizada.
    """ 
    import matplotlib.pyplot as plt
    
    # Cargar la imagen
    try:
        imagen_original = plt.imread('lena.pgm')
        if imagen_original.max() <= 1.0:
            imagen_original = imagen_original * 255.0
    except FileNotFoundError:
        print("Aviso: No se encontró 'lena.pgm'. Generando imagen de prueba con ruido...")
        imagen_original = np.random.randint(0, 256, (128, 128))

    ruido = np.random.normal(0, 25, imagen_original.shape)
    imagen_ruidosa = np.clip(imagen_original + ruido, 0, 255)
  
    # Lambda de prueba
    f = Function(imag_ori=imagen_ruidosa, lmbda=0.5)

    # Descenso de Gradiente para suavizar la imagen
    print("Suavizando imagen...")
    x_actual = imagen_ruidosa.flatten()
    alpha = 0.05  # Tamaño de paso
    
    for iteracion in range(50):
        grad = f.diff(x_actual)
        x_actual = x_actual - alpha * grad
        
        if (iteracion+1) % 10 == 0:
            print(f"Iteración {iteracion+1}/50 completada")

    imagen_suavizada = x_actual.reshape(f.forma)

    # Generar original vs suavizada
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
