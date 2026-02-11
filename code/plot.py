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


    # LO NUEVO
    def draw_contours(self, function_obj, range_val=[-10, 10], density=100): # <--- NUEVO
        """
        Dibuja los círculos de nivel (contour) de la función que le pases.
        """
        if self.ax is None:
            print("¡Error! Primero tienes que llamar a .canvas()")
            return

        # 1. Preparar el piso (Meshgrid)
        x = np.linspace(range_val[0], range_val[1], density)
        y = np.linspace(range_val[0], range_val[1], density)
        X, Y = np.meshgrid(x, y)

        # 2. Calcular alturas (Z) usando la función que te pasen
        # (Aquí asumimos que function_obj tiene un método .eval())
        Z = np.zeros_like(X)

        # Un doble for simple para asegurar que funcione con cualquier función
        for i in range(len(x)):
            for j in range(len(y)):
                # Creamos el punto [x, y]
                point = np.array([X[i, j], Y[i, j]])
                # Le preguntamos la altura a la función
                Z[i, j] = function_obj.eval(point)

        # 3. Dibujar los contornos
        # levels=20 son los círculos, cmap es el color
        contour = self.ax.contour(X, Y, Z, levels=20, cmap='viridis')
        
        # Opcional: Agregar una barra de color para saber qué altura es
        self.fig.colorbar(contour, ax=self.ax) 
    # ---------------------------------------------------------

    def show(self):
        """  Docstring for show  """
        plt.show()

# --- CLASE DUMMY PARA PROBAR (SOLO PARA QUE VEAS QUE FUNCIONA) ---
class DummySphere:
    def eval(self, x):
        return x[0]**2 + x[1]**2  # x^2 + y^2

def main():

    # aqui probar la clase

    my_plot = Plot('Mapa de niveles (prueba)')
    my_plot.canvas()

    esfera_falsa = DummySphere()

    my_plot.draw_contours(esfera_falsa)
    
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
