"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 15/04/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA: Optimización
DESCRIPTION:
    Registro afín de imágenes usando BFGS con función objetivo SSD.
    Imágenes fija: I_1.pgm (referencia)  |  Móvil: I_6.pgm

    Estrategia de optimización (3 pasos):
      1. Grid search exhaustivo a 1/4 de resolución → punto inicial global
      2. Refinamiento L-BFGS-B a 1/2 de resolución
      3. Refinamiento L-BFGS-B a resolución completa
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates, zoom

DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xs_04.func_obj import AffineRegistrationObjective
from xs_04.modelo import AffineModel6
from xs_04.plot_registration import RegistrationPlot


# ── Configuración ─────────────────────────────────────────────────────────────
FIXED_PATH  = os.path.join(DIR, "I_1.pgm")
MOVING_PATH = os.path.join(DIR, "I_6.pgm")


# ── Utilidades ────────────────────────────────────────────────────────────────
def cargar_gray(path: str) -> np.ndarray:
    img = plt.imread(path).astype(float)
    if img.ndim == 3:
        img = img[..., 0]
    vmax = img.max()
    return img / vmax if vmax > 0 else img


def _refinar(func, theta_init, gtol=1e-7):
    """Refina theta con L-BFGS-B sin bounds."""
    res = minimize(func.eval, theta_init, jac=func.diff, method='L-BFGS-B',
                   options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': gtol})
    return res.x, res.fun, res.nit


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Registro afín de imágenes — BFGS (SSD) ===\n")

    # 1. Cargar imágenes normalizadas
    fixed_full  = cargar_gray(FIXED_PATH)
    moving_full = cargar_gray(MOVING_PATH)
    h, w = fixed_full.shape
    print(f"I_1 (fija)  shape: {fixed_full.shape}")
    print(f"I_6 (móvil) shape: {moving_full.shape}\n")

    f0 = AffineRegistrationObjective(fixed_full, moving_full).eval(AffineModel6.identity_theta())
    print(f"f(identidad) = {f0:.4f}\n")

    # ── Paso 1: Grid search a 1/4 de resolución ──────────────────────────────
    print("Paso 1: Grid search (1/4 resolución)...")
    scale = 0.25
    fixed_s  = zoom(fixed_full,  scale, order=1)
    moving_s = zoom(moving_full, scale, order=1)
    hs, ws   = fixed_s.shape

    i_c = np.arange(hs)[:, None].astype(float)
    j_c = np.arange(ws)[None, :].astype(float)

    best_f   = 1e18
    best_t   = AffineModel6.identity_theta()

    for t1 in np.linspace(0.7, 1.3, 7):
     for t5 in np.linspace(0.7, 1.3, 7):
      for t2 in np.linspace(-0.3, 0.3, 5):
       for t4 in np.linspace(-0.3, 0.3, 5):
        for t3 in np.linspace(-hs*0.4, hs*0.4, 9):
         for t6 in np.linspace(-ws*0.4, ws*0.4, 9):
          rw = t1*i_c + t2*j_c + t3
          cw = t4*i_c + t5*j_c + t6
          warped = map_coordinates(moving_s, [rw, cw], order=1,
                                   mode='constant', cval=0.0)
          f = 0.5 * np.sum((fixed_s - warped) ** 2)
          if f < best_f:
              best_f = f
              best_t = np.array([t1, t2, t3, t4, t5, t6])

    # Escalar traslaciones al tamaño completo
    theta_init = best_t.copy()
    theta_init[2] /= scale   # t3 fila
    theta_init[5] /= scale   # t6 col
    print(f"  Mejor inicial: f={best_f:.4f}  theta={np.round(best_t,3)}")
    print(f"  (trasl. a tamaño completo: t3={theta_init[2]:.1f}, t6={theta_init[5]:.1f})\n")

    # ── Paso 2: Refinamiento a 1/2 resolución ────────────────────────────────
    print("Paso 2: Refinamiento (1/2 resolución)...")
    scale = 0.5
    fixed_h  = zoom(fixed_full,  scale, order=1)
    moving_h = zoom(moving_full, scale, order=1)
    func_h   = AffineRegistrationObjective(fixed_h, moving_h)

    theta_h = theta_init.copy()
    theta_h[2] *= scale; theta_h[5] *= scale

    theta_h, fh, nit_h = _refinar(func_h, theta_h)
    theta_h[2] /= scale; theta_h[5] /= scale
    print(f"  f={fh:.4f}  iters={nit_h}  theta={np.round(theta_h,4)}\n")

    # ── Paso 3: Refinamiento a resolución completa ───────────────────────────
    print("Paso 3: Refinamiento (resolución completa)...")
    func_full = AffineRegistrationObjective(fixed_full, moving_full)

    theta_opt, f_opt, nit_f = _refinar(func_full, theta_h)
    print(f"  f={f_opt:.4f}  iters={nit_f}  theta={np.round(theta_opt,5)}\n")

    print("─" * 50)
    print(f"SSD: {f0:.2f} → {f_opt:.2f}  (reducción {(1-f_opt/f0)*100:.1f}%)")
    labels = ["t1 (escala fila)", "t2 (cizalla)", "t3 (trasl. fila)",
              "t4 (cizalla)",    "t5 (escala col)", "t6 (trasl. col)"]
    print("\nTheta óptimo:")
    for lb, v in zip(labels, theta_opt):
        print(f"  {lb}: {v:.6f}")

    # ── Visualizar ────────────────────────────────────────────────────────────
    plotter = RegistrationPlot(
        title=f"Registro afín — BFGS  (SSD: {f0:.0f} → {f_opt:.0f}, {(1-f_opt/f0)*100:.1f}%)"
    )
    plotter.show(fixed_img=fixed_full, moving_img=moving_full, theta=theta_opt)


if __name__ == "__main__":
    main()