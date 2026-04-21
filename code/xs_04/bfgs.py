"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 20/04/2026
UDA: Optimización
DESCRIPTION: BFGS con punto inicial aleatorio de gran escala (-100 a 100).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class BFGSOptimizer:
    def __init__(self, func, max_it=2000, tolerance=1e-7, alpha0=1.0, rho=0.5):
        self.func = func
        self.max_it = max_it
        self.tolerance = tolerance
        self.alpha0 = alpha0
        self.rho = rho
        self.path = []

    def line_search(self, xk, pk):
        alpha = float(self.alpha0)
        fk = self.func.eval(xk)
        gk = self.func.diff(xk)
        while True:
            xk_next = xk + alpha * pk
            if self.func.eval(xk_next) <= fk + 1e-4 * alpha * np.dot(gk, pk):
                break
            alpha *= self.rho
            if alpha < 1e-16: break
        return alpha

    def solve(self, x0):
        xk = np.array(x0, dtype=float).ravel()
        n = len(xk)
        Hk = np.eye(n)
        self.path = [] 
        
        for i in range(self.max_it):
            self.path.append(xk.copy())
            gk = self.func.diff(xk)
            if np.linalg.norm(gk) < self.tolerance:
                break

            pk = -(Hk @ gk)
            if np.dot(gk, pk) > 0:
                Hk = np.eye(n)
                pk = -gk

            alpha = self.line_search(xk, pk)
            sk = alpha * pk
            xk_next = xk + sk
            
            yk = self.func.diff(xk_next) - gk
            ys = np.dot(yk, sk)
            
            if abs(ys) > 1e-12:
                rho_k = 1.0 / ys
                I = np.eye(n)
                A1 = I - rho_k * np.outer(sk, yk)
                A2 = I - rho_k * np.outer(yk, sk)
                Hk = A1 @ Hk @ A2 + (rho_k * np.outer(sk, sk))
            else:
                Hk = np.eye(n)
            
            xk = xk_next
        return xk
class RosenbrockND:
    def eval(self, x):
        x = np.asarray(x).ravel()
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))
        
    def diff(self, x):
        x = np.asarray(x).ravel()
        grad = np.zeros_like(x)
        if len(x) < 2: return grad
        grad[:-1] += -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad

def graficar_proyeccion_2d(optimizer, func, x0):
    x_range = np.linspace(-110, 110, 150)
    y_range = np.linspace(-110, 110, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            Z[j, i] = 100 * (Y[j, i] - X[j, i]**2)**2 + (1 - X[j, i])**2

    path = np.array(optimizer.path)

    plt.figure(figsize=(12, 8))
    levels = np.logspace(-0.5, 9, 30)
    plt.contourf(X, Y, Z, levels=levels, cmap='magma', alpha=0.6)
    
    plt.plot(path[:, 0], path[:, 1], color='white', linewidth=3, alpha=0.5, zorder=3)
    plt.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=3, 
             linewidth=1, label='Trayectoria BFGS (Proyección x1, x2)', zorder=4)
    
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Inicio Aleatorio', zorder=5)
    plt.plot(1, 1, 'cyan', marker='*', markersize=18, label='Óptimo (1,1)', zorder=5, markeredgecolor='black')

    plt.title(f'Optimización BFGS desde punto aleatorio [-100, 100]\nDimensiones totales: {len(x0)} | Pasos: {len(path)}', fontsize=13)
    plt.xlabel('Variable x1')
    plt.ylabel('Variable x2')
    plt.legend()
    plt.colorbar(label='f(x)')
    plt.show()


if __name__ == "__main__":
    f_nd = RosenbrockND()
    opt = BFGSOptimizer(f_nd)
    
    dimensiones = 20
    x0_aleatorio = np.random.uniform(-100, 100, dimensiones)
    
    print(f"--- INICIANDO PRUEBA EN {dimensiones}D ---")
    print(f"Punto inicial (primeros 5): {x0_aleatorio[:5]}")
    
    solucion = opt.solve(x0_aleatorio)
    
    print("-" * 40)
    print(f"Finalizado en {len(opt.path)} iteraciones.")
    print(f"Distancia final al óptimo: {np.linalg.norm(solucion - 1):.2e}")
    
    graficar_proyeccion_2d(opt, f_nd, x0_aleatorio)