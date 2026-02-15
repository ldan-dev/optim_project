"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
Implementación de la función de Rosenbrock
"""
import numpy as np
from function import Function
from plot import Plot

class Func_Rosen(Function):
    """
    Rosenbrock function
    """
    def __init__(self, params=None):
        """constructor"""
        super().__init__(params)
        self.name = "Rosenbrock"
    
    def eval(self, x: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function at point x        
        """
        x = np.asarray(x)
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("x must have numerical elements")
        
        # F(x) = sum_{i=1}^{d-1} [(1-x_i)^2 + 100(x_{i+1}-x_i^2)^2]
        result = 0.0
        for i in range(len(x) - 1):
            result += (1 - x[i])**2 + 100 * (x[i+1] - x[i]**2)**2
        
        return result
    

    def diff(self, x: np.ndarray) -> np.ndarray:
        """
        return the gradient vector at x (1st derivative)
        """
        x = self._Function__validate_x(x)
        d = len(x)
        grad = np.zeros(d)
        
        if d == 1:
            return grad
        
        # i=1
        grad[0] = 2*(x[0] - 1) - 400*x[0]*(x[1] - x[0]**2)
        
        # i = 2, 3, ... , n 
        for i in range(1, d-1):
            grad[i] = 2*(x[i] - 1) + 200*(x[i] - x[i-1]**2 - 2*x[i]*(x[i+1] - x[i]**2) )
        
        # i = n
        grad[d-1] = 200*(x[d-1] - x[d-2]**2)
        
        return grad
    
    
    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """
        return the Hessian matrix at x (2nd derivative) 
        """
        x = self._Function__validate_x(x)
        d = len(x)
        H = np.zeros((d, d))
        
        # Based on your notes, the Hessian has a tridiagonal-like structure
        # Diagonal elements
        for i in range(d):
            if i == 0:
                H[i, i] = 2 + 1200*x[i]**2 - 400*x[i+1]
            elif i == d-1:
                H[i, i] = 200
            else:
                H[i, i] = 2 + 200 + 1200*x[i]**2 - 400*x[i+1]
        
        # Off-diagonal elements
        for i in range(d-1):
            H[i, i+1] = -400*x[i]
            H[i+1, i] = -400*x[i]
        
        return H
    
    def plot_2d(self, lim: list[float], canva: Plot):
        """
        Plot the Rosenbrock function in 2D
        
        Parameters:
        -----------
        lim : list[float]
            Plot limits [xmin, xmax, ymin, ymax]
        canva : Plot
            Plot canvas object
        """
        if len(lim) != 4:
            raise ValueError("lim must have 4 elements [xmin, xmax, ymin, ymax]")
        
        x = np.linspace(lim[0], lim[1], 100)
        y = np.linspace(lim[2], lim[3], 100)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.eval(point)
        
        canva.contour(X, Y, Z, levels=50)


def main():
    f = Func_Rosen()
    
    # x_test = np.array([1.0, 1.0])
    
    # Test gradient
    x_test2 = np.array([10.0, 10.0])
    print(f"F({x_test2}) = {f.eval(x_test2)}")
    
    print(f"∇F({x_test2}) = {f.diff(x_test2)}")
    
    # Test Hessian
    print(f"∇²F({x_test2}) =\n{f.ddiff(x_test2)}")
    
    # # Test plot
    # print("\nGenerando gráfica 2D de la función de Rosenbrock...")
    # plot_obj = Plot("Función de Rosenbrock")
    # plot_obj.canvas(xlabel='x1', ylabel='x2')
    
    # # Límites para visualizar bien la función (el mínimo está en [1,1])
    # lim = [-2, 2, -1, 3]  # [xmin, xmax, ymin, ymax]
    # f.plot_2d(lim, plot_obj)
    
    # # Marcar el mínimo global en [1, 1]
    # plot_obj.ax.plot(1, 1, 'r*', markersize=15, label='Mínimo global (1,1)')
    # plot_obj.ax.legend()
    
    # plot_obj.show()

if __name__ == "__main__":
    main()