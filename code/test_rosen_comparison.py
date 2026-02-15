import numpy as np
import matplotlib.pyplot as plt

class Rosenbrock:
    def __init__(self, a=1, b=100):
        """
        Inicializa los parámetros de la función. 
        Por defecto a=1, b=100 (valores estándar).
        """
        self.a = a
        self.b = b

    def eval(self, x: np.ndarray) -> float:
        """Calcula el valor de la función para un vector x de N dimensiones."""
        x = np.asarray(x)
        # Sumatoria de b*(x_{i+1} - x_i^2)^2 + (a - x_i)^2
        return np.sum(self.b * (x[1:] - x[:-1]**2)**2 + (self.a - x[:-1])**2)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """Calcula el gradiente (vector de 1eras derivadas)."""
        x = np.asarray(x)
        n = len(x)
        grad = np.zeros_like(x)
        
        # Derivadas parciales generalizadas para N dimensiones
        grad[:-1] += -4 * self.b * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (self.a - x[:-1])
        grad[1:] += 2 * self.b * (x[1:] - x[:-1]**2)
        
        return grad

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """Calcula la matriz Hessiana (2das derivadas)."""
        x = np.asarray(x)
        n = len(x)
        hess = np.zeros((n, n))
        
        # Diagonal principal
        # Para i desde 0 hasta n-2
        hess[np.arange(n-1), np.arange(n-1)] = 12 * self.b * x[:-1]**2 - 4 * self.b * x[1:] + 2
        # Para i desde 1 hasta n-1
        hess[np.arange(1, n), np.arange(1, n)] += 2 * self.b
        
        # Diagonales secundarias (i, i+1) e (i+1, i)
        indices = np.arange(n-1)
        hess[indices, indices + 1] = -4 * self.b * x[:-1]
        hess[indices + 1, indices] = -4 * self.b * x[:-1]
        
        return hess

    def plot_2d(self, x_range=(-2, 2), y_range=(-1, 3), res=100):
        """Genera una visualización de contorno y superficie 3D."""
        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        
        # Evaluación para el grid
        Z = self.b * (Y - X**2)**2 + (self.a - X)**2
        
        fig = plt.figure(figsize=(14, 6))
        
        # Mapa de contorno
        ax1 = fig.add_subplot(121)
        cp = ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='magma')
        ax1.plot(self.a, self.a**2, 'r*', markersize=15, label='Mínimo Global')
        ax1.set_title('Curvas de Nivel (Vista Superior)')
        ax1.legend()
        plt.colorbar(cp, ax=ax1)
        
        # Superficie 3D
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, Z, cmap='magma', alpha=0.8, edgecolor='none')
        ax2.set_title('Superficie de Rosenbrock')
        
        plt.show()

if __name__ == "__main__":
    # Instanciar la función
    rosen = Rosenbrock()

    # Evaluar en un punto específico (ej: x=0, y=0)
    punto = np.array([10.0, 10.0])
    print(f"Valor en {punto}: {rosen.eval(punto)}")
    print(f"Gradiente en {punto}: {rosen.diff(punto)}")
    print(f"Hessiana en {punto}:\n{rosen.ddiff(punto)}")

    # Graficar
    rosen.plot_2d()
        