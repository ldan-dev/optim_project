import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from conjugate_gradient import ConjugateGradient
import scipy.sparse as sp
import os
import sys

ruta_actual = os.getcwd()
if "code" not in ruta_actual:
    sys.path.append(os.path.join(ruta_actual, 'code'))

class FuncImageSmoothing:
    def __init__(self, image_array, lmbda=0.5):
        self.b = image_array.flatten()
        self.shape = image_array.shape
        n = len(self.b)
        
        I = sp.eye(n, format='csr')
        offsets = [-1, 0, 1]
        data = np.array([-np.ones(n), 2*np.ones(n), -np.ones(n)])
        L = sp.dia_matrix((data, offsets), shape=(n, n)).tocsr()
        
        self.A = I + lmbda * L

    def diff(self, x):
        return (self.A @ x) - self.b

    def ddiff(self, x):
        return self.A

def main():
    archivo_entrada = os.path.join("code", "lena.pgm")
    archivo_salida = os.path.join("code", "lena_resultado.png")

    if not os.path.exists(archivo_entrada):
        print(f"ERROR: No se encuentra '{archivo_entrada}'")
        print(f"Archivos que veo aqui: {os.listdir()}")
        if os.path.exists("code"):
            print(f"Archivos dentro de 'code': {os.listdir('code')}")
        return

    try:

        print(f"Cargando {archivo_entrada}...")
        img_raw = Image.open(archivo_entrada).convert('L')
        
        img_resized = img_raw.resize((512, 512))
        img_np = np.array(img_resized, dtype=float) / 255.0
        
        problema = FuncImageSmoothing(img_np, lmbda=0.01)
        

        cg = ConjugateGradient(func=problema, max_it=100, tolerance=1e-5)
        
        print("Iniciando optimizacion...")
        resultado_vec = cg.solve(start_point=img_np.flatten(), verbose=True)
        
        # 4. Reconstruccion
        img_final = resultado_vec.reshape(img_np.shape)
        
        # 5. Guardar resultado
        img_guardar = Image.fromarray((img_final * 255).astype(np.uint8))
        img_guardar.save(archivo_salida)
        print(f"¡Exito! Resultado guardado en: {archivo_salida}")
        
        # 6. Mostrar comparativa en pantalla
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Original (Lena PGM)")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_final, cmap='gray')
        plt.title("Suavizada (CG)")
        plt.show()

    except Exception as e:
        print(f"Ocurrio un error critico: {e}")

if __name__ == "__main__":
    main()