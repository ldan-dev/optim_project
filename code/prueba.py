import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from conjugate_gradient import ConjugateGradient

class FuncImage:
    def __init__(self, b_noisy, lmbda=0.5):
        self.shape = b_noisy.shape
        self.b = b_noisy.flatten()
        self.lmbda = lmbda
        n = len(self.b)
        # Matriz A: Identidad + lambda * Laplaciano
        # Esto mantiene la imagen original pero penaliza ruido
        self.A = np.eye(n) + self.lmbda * self._generate_laplacian(n)

    def _generate_laplacian(self, n):
        L = np.eye(n) * 2
        for i in range(n-1):
            L[i, i+1] = L[i+1, i] = -1
        return L

    def diff(self, x): 
        return (self.A @ x) - self.b
        
    def ddiff(self, x): 
        return self.A

def main():
    # 1. Crear una imagen de prueba simple (un cuadro blanco en fondo negro)
    img_size = 30
    img = np.zeros((img_size, img_size))
    img[10:20, 10:20] = 1.0
    
    # 2. Agregar ruido aleatorio
    noise = np.random.normal(0, 0.2, img.shape)
    img_noisy = img + noise
    
    # 3. Configurar optimizador
    f_img = FuncImage(img_noisy, lmbda=1.5)
    cg = ConjugateGradient(func=f_img, max_it=100, tolerance=1e-4)
    
    # 4. Resolver
    print("Suavizando imagen con Gradiente Conjugado...")
    res_vec = cg.solve(start_point=img_noisy.flatten(), verbose=True)
    img_final = res_vec.reshape(img.shape)
    
    # 5. Mostrar resultados
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_noisy, cmap='gray')
    axs[0].set_title("Imagen con Ruido (b)")
    
    axs[1].imshow(img_final, cmap='gray')
    axs[1].set_title("Resultado Suavizado (x)")
    
    plt.show()

if __name__ == "__main__":
    main()