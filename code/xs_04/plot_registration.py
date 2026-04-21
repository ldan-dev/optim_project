"""
Visualización base para registro afín:
- Imagen fija
- Imagen móvil
- Imagen móvil transformada
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


class RegistrationPlotter:
    def __init__(self, title = "Resultados del Registro Afín", figsize = (15, 5)):
        self.title = title
        self.figsize = figsize
        self.fig = None
        self.axes = None

    def show_registration_results(self, img_fixed, img_moving, img_resultant, 
                                  title1="I_1 (Original)", 
                                  title2="I_2 (Transformada)", 
                                  title3="Corregida"):
        """
        Crea una visualización de tres imágenes una al lado de la otra.
        
        Args:
            img_fixed (numpy.ndarray): Matriz de la imagen de referencia.
            img_moving (numpy.ndarray): Matriz de la imagen distorsionada.
            img_resultant (numpy.ndarray): Matriz de la imagen móvil corregida.
        """
        self.fig, self.axes = plt.subplots(1, 3, figsize=self.figsize)
        self.fig.suptitle(self.title, fontsize=16)

        images = [img_fixed, img_moving, img_resultant]
        titles = [title1, title2, title3]

        for i in range(3):
            self.axes[i].imshow(images[i], cmap='gray')
            self.axes[i].set_title(titles[i])
            self.axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def show(self, fixed_img: np.ndarray, moving_img: np.ndarray,
             theta: np.ndarray = None, warped_img: np.ndarray = None):
        """
        Método conveniente usado por xs04_main.py.
        Si warped_img es None pero theta es dado, aplica el warp.
        """
        if warped_img is None and theta is not None:
            from xs_04.modelo import AffineModel6
            theta = AffineModel6.validate_theta(theta)
            t1, t2, t3, t4, t5, t6 = theta
            h, w = moving_img.shape
            # Convención: i=fila (eje-0), j=columna (eje-1)
            # fila* = t1*i + t2*j + t3
            # col*  = t4*i + t5*j + t6
            i_c, j_c = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            row_warp = t1 * i_c + t2 * j_c + t3
            col_warp = t4 * i_c + t5 * j_c + t6
            coords = np.stack([row_warp, col_warp])
            warped_img = map_coordinates(moving_img.astype(float), coords,
                                         order=1, mode='constant', cval=0.0)

        if warped_img is None:
            warped_img = moving_img

        self.show_registration_results(
            img_fixed=fixed_img,
            img_moving=moving_img,
            img_resultant=warped_img,
        )


# Alias para compatibilidad con xs04_main.py
RegistrationPlot = RegistrationPlotter


# Función para cargar las imágenes
def load_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: El archivo '{file_path}' no existe.")
    
    # Recibe una imagen, devuelve una matriz
    image_matrix = plt.imread(file_path)
    return image_matrix


# *** Código de prueba ***
def main():
    print("** Prueba de Visualización para Registro de Imágenes Médicas **")

    # Nombres de los archivos
    filename_fixed = "I_1.pgm"
    filename_moving = "I_6.pgm"

    try:
        print(f"Cargando {filename_fixed}...")
        img_fixed = load_image(filename_fixed)
        print(f"[{filename_fixed}] cargada exitosamente. Dimensiones de la matriz: {img_fixed.shape}")

        print(f"Cargando {filename_moving}...")
        img_moving = load_image(filename_moving)
        print(f"[{filename_moving}] cargada exitosamente. Dimensiones de la matriz: {img_moving.shape}")

        # Imagen resultante (en este caso la misma q la original)
        print("Simulando imagen resultante (placeholder)...")
        img_resultant = img_fixed.copy() 

        plotter = RegistrationPlotter(title="Prueba de Visualización de Registro de Imagen")

        print("Generando visualización...")
        plotter.show_registration_results(img_fixed, img_moving, img_resultant)

    except FileNotFoundError as err:
        print(err)
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer los archivos: {e}")
        print("Asegúrate de que 'I_1.pgm' e 'I_6.pgm' son archivos de imagen válidos.")

if __name__ == "__main__":
    main()
