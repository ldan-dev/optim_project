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
    def __init__(self, title:str):
        """
        docstring
        """
        self.fig = None
        self.ax = None
        self.title = title

    def canvas(self):
        """  Docstring for canvas  """
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title(self.title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        return self.fig, self.ax

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