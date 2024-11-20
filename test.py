import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Abrir un cuadro de diálogo para seleccionar la imagen
Tk().withdraw()
ruta_imagen = askopenfilename(title="Seleccionar una imagen", filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg;*.bmp")])

# Verificar si se seleccionó un archivo
if not ruta_imagen:
    print("No se seleccionó ninguna imagen. Saliendo del programa.")
    exit()

# Cargar la imagen
imagen = io.imread(ruta_imagen)

# Convertir a escala de grises
imagen_gris = color.rgb2gray(imagen)

# Aplicar detección de bordes (Sobel)
bordes_sobel = filters.sobel(imagen_gris)

# Aplicar detección de bordes (Canny)
bordes_canny = feature.canny(imagen_gris, sigma=1)

# Mostrar los resultados
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(imagen, cmap='gray')
axes[0].set_title('Imagen Original')
axes[0].axis('off')

axes[1].imshow(bordes_sobel, cmap='gray')
axes[1].set_title('Bordes Detectados (Sobel)')
axes[1].axis('off')

axes[2].imshow(bordes_canny, cmap='gray')
axes[2].set_title('Bordes Detectados (Canny)')
axes[2].axis('off')

plt.show()
