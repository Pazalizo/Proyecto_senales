import cv2
import numpy as np
from sklearn.svm import SVC
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

global label_ruta

# Función para extraer características de una imagen
def reconocer_figuras():

    def extract_features(image):
    # Convertir la imagen a escala de grises
        gray = np.mean(image, axis=2, dtype=np.uint8)

    # Cálculo de gradientes
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Gx = cv2.filter2D(gray, -1, kernel_x)
        Gy = cv2.filter2D(gray, -1, kernel_y)

    # Cálculo de magnitud de gradiente
        Gt = np.sqrt(Gx*2 + Gy*2)

    # Normalización de Gt
        GtN = (Gt / np.max(Gt)) * 255
        GtN = GtN.astype(np.uint8)

    # Cálculo de offset y normalización de Gx y Gy
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

    # Binarización de GtN
        _, B = cv2.threshold(GtN, 50, 255, cv2.THRESH_BINARY)
    #B = GtN > 25
    #B = B.astype(np.uint8) * 255

    # Calcular momentos de Hu
        moments = cv2.HuMoments(cv2.moments(B)).flatten()

        return moments

# Ruta de las imágenes de entrenamiento
    circle_path = "./circulos/*.png"
    square_path = "./cuadrados/*.png"
    triangle_path = "./triangulos/*.png"

# Leer las imágenes de entrenamiento
    circle_images = [cv2.imread(file) for file in glob.glob(circle_path)]
    square_images = [cv2.imread(file) for file in glob.glob(square_path)]
    triangle_images = [cv2.imread(file) for file in glob.glob(triangle_path)]

# Extraer características y generar etiquetas para las imágenes de entrenamiento
    X_train = []
    y_train = []

    image_sets = [circle_images, square_images, triangle_images]

    for label, image_set in enumerate(image_sets):
        for img in image_set:
            if img is not None:
                features = extract_features(img)
            
                X_train.append(features)
                y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)


# Entrenar el clasificador utilizando un modelo de SVM
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf.fit(X_train, y_train)

# Función para predecir la forma geométrica de una imagen
    def predict_shape(image):
    # Extraer características de la imagen
        features = extract_features(image)

    # Predecir la forma utilizando el clasificador entrenado
        prediction = clf.predict([features])

        return prediction[0]

# Ruta de la imagen a analizar
    global image_path

# Cargar la imagen a analizar
    image = cv2.imread(image_path)

# Predecir la forma geométrica de la imagen
    shape = predict_shape(image)
    if (shape == 0):
        shape = 'circulo'
    elif ( shape == 1):
        shape = 'cuadrado'
    elif(shape == 2):
        shape = 'triangulo' 
    else:
        shape = 'indeterminado'
# Mostrar la forma detectada
    messagebox.showinfo("Resultados de la deteccion", f"Forma detectada: {shape}")
    cv2.imshow("Imagen a detectar", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

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


def figuras(main):
    global label_ruta

    main.withdraw()

    root = tk.Toplevel()
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
    boton_realizar = tk.Button(root, text="Reconocer figura", command=lambda: reconocer_figuras())
    boton_realizar.pack(pady=10)

    boton_volver = tk.Button(root, text="Volver", command=lambda: volver(root, main))
    boton_volver.pack(pady=10)

    # Ejecutar el bucle principal de la aplicación
    root.mainloop()