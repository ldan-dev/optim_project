from functions.func_smooth import Function_Smooth
from pre_gradient_conjugate import Pre_Gradient_Conjugate
from conjugate_gradient import ConjugateGradient
from plot_image import ImagePlot, cargar_imagen

def main():
    try:
        img = cargar_imagen("lena.pgm")
        img_norm = img / 255.0 if img.max() > 1.0 else img
        problema = Function_Smooth(img_norm, lmbda=400)
        """
        print("Iniciando PCG...")
        solver_pcg = Pre_Gradient_Conjugate(problema, max_it=50)
        res_pcg = solver_pcg.solve(img_norm)
        """
        print("Iniciando CG...")
        solver_cg = ConjugateGradient(problema, max_it=10000)
        res_cg = solver_cg.solve(img_norm, verbose=True)
        """
        # 5. Generar los 2 plots solicitados
        """
        """
        # Plot 1: Original vs Pre-Gradiente Conjugado
        plotter_pcg = ImagePlot(title="Resultado Pre-Gradiente Conjugado")
        plotter_pcg.show_comparison(
            img_norm, 
            res_pcg.reshape(img.shape), 
            title1="Original", 
            title2="Mejorada (PCG)"
        )
        """
        # Plot 2: Original vs Gradiente Conjugado
        plotter_cg = ImagePlot(title="Gradiente Conjugado")
        plotter_cg.show_comparison(
            img_norm, 
            res_cg.reshape(img.shape), 
            title1="Original", 
            title2="Mejorada λ:10"
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()