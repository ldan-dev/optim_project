import numpy as np

def wolfe_strong_cond(func, xk, alpha, pk, c1=1e-4, c2=0.9):
    xk_next = xk + alpha * pk 
    fk = func.eval(xk) 
    fk_n = func.eval(xk_next) 
    grad_xk = func.diff(xk) 
    grad_next = func.diff(xk_next) 
    pendiente = np.dot(grad_xk, pk) 
    pendiente_n = np.dot(grad_next, pk) 
    cond1 =fk_n <= fk + c1 * alpha *  pendiente 
    cond2 = np.abs(pendiente_n) <= c2 * np.abs(pendiente) 

    return bool(cond1 and cond2) 

