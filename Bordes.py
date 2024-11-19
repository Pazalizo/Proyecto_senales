import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

global image_path

def BordesImagen():
    global image_path
    I = cv2.imread(image_path)
    Im = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    m, n = Im.shape

    Im = Im.astype(float)

    Gx = np.zeros_like(Im)
    Gy = np.zeros_like(Im)

    for r in range(1, m - 1):
        for c in range(1, n - 1):
            Gx[r, c] = -1 * Im[r - 1, c - 1] - 2 * Im[r - 1, c] - Im[r - 1, c + 1] + Im[r + 1, c - 1] + 2 * Im[r + 1, c] + Im[r + 1, c + 1]
            Gy[r, c] = -1 * Im[r - 1, c - 1] + Im[r - 1, c + 1] - 2 * Im[r, c - 1] + 2 * Im[r, c + 1] - Im[r + 1, c - 1] + Im[r + 1, c + 1]

    Gt = np.sqrt(Gx ** 2 + Gy ** 2)

    VmaxGt = np.max(Gt)

    Gtn = (Gt / VmaxGt) * 255

    Gtn = Gtn.astype(np.uint8)

    VminGx = np.min(Gx)
    VminGy = np.min(Gy)

    GradOffx = Gx - VminGx
    GradOffy = Gy - VminGy

    VmaxGx = np.max(GradOffx)
    VmaxGy = np.max(GradOffy)

    GxN = (GradOffx / VmaxGx) * 255
    GyN = (GradOffy / VmaxGy) * 255

    GxN = GxN.astype(np.uint8)
    GyN = GyN.astype(np.uint8)

    B = Gtn > 25
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(GxN,cmap='gray')
    axs[0].set_title('GxN')
    axs[0].axis('off')

    # Muestra la imagen comprimida como estática
    axs[1].imshow(GyN, cmap='gray')
    axs[1].set_title('GyN')
    axs[1].axis('off')

    # Muestra la imagen descomprimida y restaurada
    axs[2].imshow(Gtn,cmap='gray')
    axs[2].set_title('Gtn')
    axs[2].axis('off')

    axs[3].imshow(B.astype(np.uint8) * 255,cmap='gray')
    axs[3].set_title('Borde')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()

def seleccionar_imagen():
    global image_path
    global label_ruta
    image_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if image_path:
        label_ruta.config(text=f"Imagen seleccionada: {image_path}")
    else:
        label_ruta.config(text="No se ha seleccionado ninguna imagen.")

def volver(root, main):
    root.destroy()
    main.deiconify()

def bordes(main):
    global label_ruta

    main.withdraw()

    root = tk.Tk()
    root.title("Interfaz de Selección de Imagen")

    # Variable global para almacenar la ruta de la imagen
    image_path = None

    # Crear y colocar el botón para seleccionar la imagen
    boton_seleccionar = tk.Button(root, text="Seleccionar Imagen", command=seleccionar_imagen)
    boton_seleccionar.pack(pady=10)

    # Etiqueta para mostrar la ruta de la imagen seleccionada
    label_ruta = tk.Label(root, text="No se ha seleccionado ninguna imagen.")
    label_ruta.pack(pady=10)

    # Crear y colocar el botón para realizar la función
    boton_realizar = tk.Button(root, text="Reconocer bordes", command=lambda: BordesImagen())
    boton_realizar.pack(pady=10)

    boton_volver = tk.Button(root, text="Volver", command=lambda: volver(root, main))
    boton_volver.pack(pady=10)

    # Ejecutar el bucle principal de la aplicación
    root.mainloop()