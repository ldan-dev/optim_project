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
        esta clase debe de poder
        """
        self.fig = None
        self.ax = None
        self.title = title
        self.figsize = figsize

    def canvas(self, xlabel='x', ylabel='y'):
        """  Docstring for canvas  """
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title(self.title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        return self.fig, self.ax

    def contour(self, X, Y, Z, levels=50):
        """
        Create contour plot
        
        Parameters:
        -----------
        X, Y : np.ndarray
            Meshgrid arrays for x and y coordinates
        Z : np.ndarray
            Function values at each (x, y) point
        levels : int
            Number of contour levels
        """
        if self.ax is None:
            self.canvas()
        
        contour = self.ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        self.ax.clabel(contour, inline=True, fontsize=8)
        contour_filled = self.ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        self.fig.colorbar(contour_filled, ax=self.ax)
    
    def show(self):
        """  Docstring for show  """
        plt.show()

def main():

    # aqui probar la clase

    my_plot = Plot('titulo')
    my_plot.canvas()
    my_plot.show()

if __name__ == "__main__":
    main()

# path = np.array(self.path)
# # Dibuja la línea con X en cada paso (como en tu dibujo azul)
# plot_obj.ax.plot(path[:, 0], path[:, 1], 'r-x', label='Descenso', markersize=5)

# # Marcar inicio y fin
# plot_obj.ax.plot(path[0, 0], path[0, 1], 'bo', label='Inicio') # Azul
# plot_obj.ax.plot(path[-1, 0], path[-1, 1], 'go', label='Fin')  # Verde
# plot_obj.ax.legend()