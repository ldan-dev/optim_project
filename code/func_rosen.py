"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 11/02/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: 
DESCRIPTION: 
Implementacion de la funcion de Rosenbrock
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
        x = self._Function__validate_x(x)
        
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
        
        # i = 2, 3, ... , n - 1
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
        H = np.zeros((d, d)) # dxd fill of 0's
        
        diag = np.zeros(d) # main diagonal (k=0)
        diag[0] = 1200 * x[0]**2 - 400 * x[1] + 2 # (1,1) element
        if d > 2:
            diag[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        diag[-1] = 200 #(d,d) element
        
        # off-diagonal elements, elements who hug the main diagonal
        off_diag = -400 * x[:-1] # [:1] -> everyone but last
        
        # hesian
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        # k=1 -> superior diagonal
        # k=-1 -> inferior diagonal
    
        return H

        # noob version:
        # H = np.zeros((d, d))
        # for i in range(d): // diagonal elements
        #     if i == 0:
        #         H[i, i] = 2 + 1200*x[i]**2 - 400*x[i+1]
        #     elif i == d-1:
        #         H[i, i] = 200
        #     else:
        #         H[i, i] = 2 + 200 + 1200*x[i]**2 - 400*x[i+1]
        
        # # off-diagonal elements, elements who hug the main diagonal
        # for i in range(d-1):
        #     H[i, i+1] = -400*x[i]
        #     H[i+1, i] = -400*x[i]
        
        # return H
    
    def plot_2d(self, range_val=[-2, 2], density=200):
        """
        Grafica la función de Rosenbrock en 2D usando contornos.
        El mínimo global está en (1, 1).
        """
        my_plot = Plot(f'Función de {self.name}')
        my_plot.canvas(xlabel='x₁', ylabel='x₂')
        my_plot.draw_contours(self, range_val=range_val, density=density)
        my_plot.show()


def main():
    f = Func_Rosen()
    
    # x_test = np.array([1.0, 1.0])
    
    x_test2 = np.array([10.0, 10.0])
    
    print(f"F({x_test2}) = {f.eval(x_test2)}")
    
    # gradient
    print(f"∇F({x_test2}) = {f.diff(x_test2)}")
    
    # hessian
    print(f"∇^2F({x_test2}) =\n{f.ddiff(x_test2)}")
    
    # Graficar la función en 2D
    f.plot_2d(range_val=[-2, 2], density=200)


if __name__ == "__main__":
    main()
