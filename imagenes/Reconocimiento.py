import cv2
import numpy as np
from sklearn.svm import SVC
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Función para extraer características de una imagen
def extract_features(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarización de la imagen utilizando el método de Otsu
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno con el área máxima
    max_contour = max(contours, key=cv2.contourArea)

    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Determinar el número de lados del polígono
    num_sides = len(approx)

    return num_sides

# Ruta de las imágenes de entrenamiento
circle_path = "circulos/*.png"
square_path = "cuadrados/*.png"
triangle_path = "triangulos/*.png"

# Leer las imágenes de entrenamiento
circle_images = [cv2.imread(file) for file in glob.glob(circle_path)]
square_images = [cv2.imread(file) for file in glob.glob(square_path)]
triangle_images = [cv2.imread(file) for file in glob.glob(triangle_path)]

# Crear un diccionario para mapear los nombres de las figuras a los índices
shape_mapping = {
    0: "Círculo",
    3: "Triángulo",
    4: "Cuadrado"
}

# Extraer características y generar etiquetas para las imágenes de entrenamiento
X_train = []
y_train = []

image_sets = [circle_images, triangle_images, square_images]

for i, image_set in enumerate(image_sets):
    for img in image_set:
        if img is not None:
            features = extract_features(img)
            X_train.append(features)
            y_train.append(i)

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

    return shape_mapping[prediction[0]]

# Ruta de la imagen a analizar
image_path = "detectar.png"

# Cargar la imagen a analizar
image = cv2.imread(image_path)

# Predecir la forma geométrica de la imagen
shape = predict_shape(image)

# Mostrar la forma detectada
print("Forma detectada:", shape)
