"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 10/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
"""


import matplotlib.pyplot as plt
import numpy as np

class Plot():
    def __init__(self, title:str, figsize=(8, 6)):
        """
        Esta clase debe de poder inicializar la figura y los ejes.
        """
        self.fig = None
        self.ax = None
        self.title = title
        self.figsize = figsize

    def canvas(self, xlabel='x', ylabel='y'):
        """  Docstring for canvas  """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        return self.fig, self.ax

    def draw_contours(self, function_obj, range_val=[-10, 10], density=100):
        """
        Dibuja los círculos de nivel (contour) de la función.
        """
        if self.ax is None:
            print("Error: No se ha mandado llamar .canvas()")
            return

        # 1. Preparar el piso con Meshgrid
        x = np.linspace(range_val[0], range_val[1], density)
        y = np.linspace(range_val[0], range_val[1], density)
        X, Y = np.meshgrid(x, y)

        # 2. Calcular alturas (Z)
        Z = np.zeros_like(X)

        # Evaluar punto a punto
        for i in range(len(x)):
            for j in range(len(y)):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = function_obj.eval(point)

        # 3. Dibujar los contornos
        contour = self.ax.contour(X, Y, >, levels = 20, cmap = 'viridis')
        # Se verifica si ya hay colorbar
        if not self.fig.axes or len(self.fig.axes) < 2:
            self.fig.colorbar(contour, ax = self.ax)

    def draw_trace(self, path_points):
        """
        Dibuja el camino sobre el plano.
        """
        if self.ax is None:
            print("Error: No se ha mandado llamar a .canvas())
            return 

        path = np.array(path_points)

        # 1. Dibujar la trayectoria
        self.ax.plot(path[:, 0], path[:, 1], 'r--', label = 'Trayectoria', alpha = 0.8)

        # 2. Inicio (azul)
        self.ax.plot(path[-1, 0], path[-1, 1], 'bo', label = "Inicio", markersize = 8)

        # 3. Fin (rojo)
        self.ax.plot(path[-1, 0], path[-1,1], 'rx', label = 'Fin', markersize = 8)

        self.ax.legend()

    def show(self):
        plt.show()

# Clase sólo para probar
class DummySphere:
    def eval(self, x):
        return x[0]**2 + x[1]**2

def main():
    print("-- Prueba de Plot --")
    my_plot = Plot('Prueba')
    my_plot.canvas()

    # Contornos
    esfera = DummySphere()
    my_plot.draw_contours(esfera)

    # Camino
    camino = [[-2, 6], [-6, 4], [-4, 2], [-2, 1], [0,0]]
    my_plot.draw_trace(camino)

    my_plot.show()


if __name__ == "__main__":
    main()
