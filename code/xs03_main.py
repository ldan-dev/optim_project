"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 13/03/2026  (dd/mm/aaaa)
MAJOR: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Email: ld.avinaneri@ugto.mx
UDA:
DESCRIPTION:
Ejercicio: suavizado de imagen PGM con Conjugate Gradient
(vanilla y preconditioned Jacobi).
"""

import os
import argparse
import numpy as np

from plot_image import ImagePlot, cargar_imagen
from conjugate_gradient import ConjugateGradientSolver
from functions import Func_Smoothing


def _to_gray(image: np.ndarray) -> np.ndarray:
	image = np.asarray(image, dtype=float)
	if image.ndim == 2:
		return image
	if image.ndim == 3:
		return image[..., 0]
	raise ValueError("Formato de imagen no soportado")


def _resolve_image_path(cli_path: str | None = None) -> str | None:
	"""Busca `lena.pgm` en rutas comunes del repo o usa ruta por argumento."""
	if cli_path is not None:
		if os.path.exists(cli_path):
			return cli_path
		print(f"Ruta indicada no existe: {cli_path}")
		return None

	here = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(here)

	candidates = [
		os.path.join(here, "lena.pgm"),
		os.path.join(project_root, "lena.pgm"),
		os.path.join(project_root, "data", "lena.pgm"),
		os.path.join(project_root, "images", "lena.pgm"),
	]

	for path in candidates:
		if os.path.exists(path):
			return path

	for root, _, files in os.walk(project_root):
		for filename in files:
			if filename.lower() == "lena.pgm":
				return os.path.join(root, filename)

	return None


def main():
	parser = argparse.ArgumentParser(description="Suavizado de Lena con CG y PCG")
	parser.add_argument("--image", type=str, default=None, help="Ruta a imagen PGM (ej. lena.pgm)")
	parser.add_argument("--lam", type=float, default=0.15, help="Parámetro lambda de suavizado")
	parser.add_argument("--max-it", type=int, default=120, help="Iteraciones máximas")
	parser.add_argument("--tol", type=float, default=1e-4, help="Tolerancia del gradiente")
	args = parser.parse_args()

	ruta_imagen = _resolve_image_path(args.image)
	lam = 0.15
	max_it = 120
	tol = 1e-4

	lam = args.lam
	max_it = args.max_it
	tol = args.tol

	if ruta_imagen is None:
		print("No se encontró 'lena.pgm'.")
		print("Opciones:")
		print("  1) Copiar `lena.pgm` dentro de `code/`")
		print("  2) Ejecutar con: python code/xs03_main.py --image /ruta/a/lena.pgm")
		return

	print(f"Usando imagen: {ruta_imagen}")

	# 1) Cargar imagen y normalizar a [0, 1]
	img_original = _to_gray(cargar_imagen(ruta_imagen))
	if img_original.max() > 1.0:
		img_original = img_original / 255.0

	# 2) Construir función cuadrática de suavizado
	smooth_func = Func_Smoothing(original_img=img_original, lam=lam)
	x0 = smooth_func.flatten_image(img_original)

	# 3) Conjugate Gradient vanilla
	cg_vanilla = ConjugateGradientSolver(max_it=max_it, tolerance=tol)
	x_vanilla = cg_vanilla.solve_vanilla(
		apply_A=smooth_func.apply_hessian,
		b=smooth_func.b,
		x0=x0,
	)

	# 4) Conjugate Gradient preconditioned (Jacobi)
	cg_prec = ConjugateGradientSolver(max_it=max_it, tolerance=tol)
	x_prec = cg_prec.solve_preconditioned(
		apply_A=smooth_func.apply_hessian,
		b=smooth_func.b,
		x0=x0,
		M_diag=smooth_func.diag,
	)

	img_suavizada = np.clip(smooth_func.unflatten_image(x_prec), 0.0, 1.0)

	print("=== Resultados Conjugate Gradient ===")
	print(f"Vanilla iteraciones: {len(cg_vanilla.path) - 1}")
	print(f"Vanilla ||g_k|| final: {cg_vanilla.residual_norms[-1]:.6e}")
	print(f"Preconditioned iteraciones: {len(cg_prec.path) - 1}")
	print(f"Preconditioned ||g_k|| final: {cg_prec.residual_norms[-1]:.6e}")

	print(f"f(x0): {smooth_func.eval(x0):.6f}")
	print(f"f(x_vanilla): {smooth_func.eval(x_vanilla):.6f}")
	print(f"f(x_preconditioned): {smooth_func.eval(x_prec):.6f}")

	plotter = ImagePlot(title="Suavizado Lena (Conjugate Gradient Preconditioned)")
	plotter.show_comparison(
		img_original,
		img_suavizada,
		title1="Original",
		title2="Suavizada (PCG-Jacobi)",
	)


if __name__ == "__main__":
	main()

