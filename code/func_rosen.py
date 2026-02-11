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
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
            
        Returns:
        --------
        float : Function value at x
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
        Compute the gradient (first derivative) of Rosenbrock function
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
            
        Returns:
        --------
        np.ndarray : Gradient vector at x
        """
        x = np.asarray(x, dtype=float)
        d = len(x)
        grad = np.zeros(d)
        
        # Based on your notes:
        # ∂F/∂x_i = -2(1-x_i) - 400x_i(x_{i+1}-x_i^2)  for i < d
        # ∂F/∂x_i = 200(x_i - x_{i-1}^2)                for i > 1
        
        # First element
        if d > 1:
            grad[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        
        # Middle elements
        for i in range(1, d-1):
            grad[i] = 200*(x[i] - x[i-1]**2) - 2*(1 - x[i]) - 400*x[i]*(x[i+1] - x[i]**2)
        
        # Last element
        if d > 1:
            grad[d-1] = 200*(x[d-1] - x[d-2]**2)
        
        return grad
    
    def ddiff(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian (second derivative) of Rosenbrock function
        
        Parameters:
        -----------
        x : np.ndarray
            Input vector
            
        Returns:
        --------
        np.ndarray : Hessian matrix at x
        """
        x = np.asarray(x, dtype=float)
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
    # Create instance
    f = Func_Rosen()
    
    # Test evaluation
    x_test = np.array([1.0, 1.0])
    print(f"F({x_test}) = {f.eval(x_test)}")
    
    # Test gradient
    x_test2 = np.array([0.0, 0.0])
    print(f"∇F({x_test2}) = {f.diff(x_test2)}")
    
    # Test Hessian
    print(f"∇²F({x_test2}) =\n{f.ddiff(x_test2)}")


if __name__ == "__main__":
    main()