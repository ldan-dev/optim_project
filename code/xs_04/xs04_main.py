"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: Optimización
DESCRIPTION:
    Registro afín de imágenes usando BFGS con función objetivo SSD.
    Imágenes fija: I_1.pgm  |  Móvil: I_6.pgm
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Rutas relativas al directorio de este archivo ---
DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xs_04.func_obj import AffineRegistrationObjective
from xs_04.bfgs import BFGSOptimizer
from xs_04.modelo import AffineModel6
from xs_04.plot_registration import RegistrationPlot


# ── Configuración ─────────────────────────────────────────────────────────────
FIXED_PATH  = os.path.join(DIR, "I_1.pgm")   # imagen de referencia
MOVING_PATH = os.path.join(DIR, "I_6.pgm")   # imagen a registrar
MAX_IT = 200
TOL    = 1e-8


# ── Utilidades ────────────────────────────────────────────────────────────────
def cargar_gray(path: str) -> np.ndarray:
    """Carga una imagen, la convierte a escala de grises y normaliza a [0, 1]."""
    img = plt.imread(path).astype(float)
    if img.ndim == 3:
        img = img[..., 0]
    elif img.ndim != 2:
        raise ValueError(f"Formato no soportado: {img.shape}")
    # Normalizar para que el gradiente tenga magnitudes razonables
    vmax = img.max()
    if vmax > 0:
        img = img / vmax
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Registro afín de imágenes — BFGS ===\n")

    # 1. Cargar imágenes
    print(f"Cargando imagen fija : {FIXED_PATH}")
    fixed_img  = cargar_gray(FIXED_PATH)
    print(f"  shape: {fixed_img.shape}")

    print(f"Cargando imagen móvil: {MOVING_PATH}")
    moving_img = cargar_gray(MOVING_PATH)
    print(f"  shape: {moving_img.shape}\n")

    # 2. Función objetivo SSD
    func = AffineRegistrationObjective(fixed_img=fixed_img, moving_img=moving_img)

    # 3. Optimizador BFGS  (alpha0 y rho ajustados para imágenes normalizadas)
    solver = BFGSOptimizer(func=func, max_it=MAX_IT, tolerance=TOL, alpha0=1e-5, rho=0.8)

    # 4. Resolver desde la identidad
    theta0 = AffineModel6.identity_theta()
    print(f"Theta inicial : {theta0}")
    print(f"f(theta0)     = {func.eval(theta0):.4f}\n")

    theta_opt = solver.solve(theta0)

    print("─" * 40)
    print("Theta óptimo encontrado:")
    labels = ["t1 (escala x)", "t2 (ciz. x←y)", "t3 (trasl. x)",
              "t4 (ciz. y←x)", "t5 (escala y)", "t6 (trasl. y)"]
    for label, val in zip(labels, theta_opt):
        print(f"  {label}: {val:.6f}")
    print(f"\nf(theta_opt)  = {func.eval(theta_opt):.4f}")
    print(f"Iteraciones   = {len(solver.path)}")

    # 5. Visualizar resultados
    plotter = RegistrationPlot(title="Registro afín — BFGS (SSD)")
    plotter.show(fixed_img=fixed_img, moving_img=moving_img, theta=theta_opt)


if __name__ == "__main__":
    main()