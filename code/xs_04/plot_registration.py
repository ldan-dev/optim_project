"""
Visualización base para registro afín:
- imagen fija
- imagen móvil
- (opcional) imagen móvil transformada
- vector theta (6 parámetros)
"""

import numpy as np
import matplotlib.pyplot as plt


class RegistrationPlot:
    def __init__(self, title: str = "Registro afín", figsize=(14, 5)):
        self.title = title
        self.figsize = figsize

    def _theta_text(self, theta: np.ndarray) -> str:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        lines = [f"theta{i+1}: {theta[i]: .5f}" for i in range(min(theta.size, 6))]
        return "\n".join(lines)

    def show(self, fixed_img: np.ndarray, moving_img: np.ndarray, theta: np.ndarray, warped_img: np.ndarray | None = None):
        fixed_img = np.asarray(fixed_img)
        moving_img = np.asarray(moving_img)

        ncols = 3 if warped_img is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=self.figsize)
        fig.suptitle(self.title, fontsize=14)

        if ncols == 2:
            ax_fixed, ax_moving = axes
        else:
            ax_fixed, ax_moving, ax_warped = axes

        ax_fixed.imshow(fixed_img, cmap="gray")
        ax_fixed.set_title("Imagen fija")
        ax_fixed.axis("off")

        ax_moving.imshow(moving_img, cmap="gray")
        ax_moving.set_title("Imagen móvil")
        ax_moving.axis("off")

        if ncols == 3:
            ax_warped.imshow(warped_img, cmap="gray")
            ax_warped.set_title("Móvil transformada")
            ax_warped.axis("off")

        fig.text(
            0.82,
            0.52,
            self._theta_text(theta),
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        plt.tight_layout()
        plt.show()
