import os
import matplotlib.pyplot as plt
import numpy as np

class ImagePlot():
  def __init__(self, title="Resultados", figsize=(10,5)):
    self.title = title
    self.figsize = figsize
    self.fig = None
    self.axes = None

  def show_comparison(self, img_original, img_suavizada, title1="Original", title2="Suavizada"):
    self.fig, self.axes = plt.subplots(1, 2, figsize = self.figsize)
    self.fig.suptitle(self.title, fontsize = 16)

    self.axes[0].imshow(img_original, cmap = 'gray')
    self.axes[0].set_title(title1)
    self.axes[0].axis('off')

    self.axes[1].imshow(img_suavizada, cmap = 'gray')
    self.axes[1].set_title(title2)
    self.axes[1].axis('off')

    plt.show()

def cargar_imagen(ruta_archivo):
  """
  Verifica que el archivo exista y carga la imagen como una matriz numpy
  """
  if not os.path.exists(ruta_archivo):
    raise FileNotFoundError(f"Error: El archivo '{ruta_archivo}' no existe. Verifique que el archivo esté en al carpeta correcta.")
  
  imagen = plt.imread(ruta_archivo)
  return imagen

def main():
  print("** Prueba de Plot_Image **")

  ruta_imagen = "lena.pgm"

  try:
    # 1. Se carga la imagen original
    img_original = cargar_imagen(ruta_imagen)

    # 2. Muestra una copia (por ahora)
    img_suavizada = img_original.copy()

    # 3. Se muestran los resultados
    mi_plot = ImagePlot(title = "Suavización de una imagen")
    mi_plot.show_comparison(img_original, img_suavizada, title1 = "Imagen original", title2 = "Imagen suavizada")

  except FileNotFoundError as err:
    print(err)

if __name__ == "__main__":
  main()
