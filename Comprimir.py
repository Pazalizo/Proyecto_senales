import sys
import os
import pyaudio
import wave
import numpy as np
from scipy.io import wavfile as waves
from scipy.fft import fft
import pickle
from scipy.spatial.distance import cosine
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget,
    QStatusBar, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import cv2
from scipy.fftpack import dct, idct


# Configuración de audio
formato = pyaudio.paInt16
canales = 1
rate = 44100
chunk = 1024
duracion = 2
archivo = "señal.wav"


class AudioProcessor(QThread):
    """Hilo para grabar y procesar comandos de audio."""
    command_detected = pyqtSignal(str)  # Señal para enviar el comando detectado

    def run(self):
        command = detect_command()
        self.command_detected.emit(command)


def record():
    """Graba un audio de 2 segundos."""
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(
            format=formato, channels=canales, rate=rate,
            input=True, input_device_index=1, frames_per_buffer=chunk
        )
    except Exception as e:
        print(f"Error abriendo el dispositivo de audio: {e}")
        return

    frames = []
    for _ in range(0, int(rate / chunk * duracion)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(canales)
    waveFile.setsampwidth(audio.get_sample_size(formato))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def detect_command():
    """Detecta el comando de audio grabado."""
    record()
    xn, energia = calculing(archivo)
    if np.max(np.abs(xn)) == 0:
        return "Desconocido"
    xn = xn / np.max(np.abs(xn))
    fft_arr = fft(xn)
    fft_arr = absFft(fft_arr)
    rec = separate(fft_arr)

    # Asegúrate de que 'array_data.pkl' exista y contenga los datos esperados
    if not os.path.exists('array_data.pkl'):
        print("El archivo 'array_data.pkl' no existe.")
        return "Desconocido"

    with open('array_data.pkl', 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, tuple) or len(data) != 2:
        print("El contenido de 'array_data.pkl' no es válido.")
        return "Desconocido"

    array, energias = data

    compare = [cosine(a, rec) for a in array]
    if not compare:
        return "Desconocido"
    pos = compare.index(min(compare))

    commands = ["Comprimir", "Segmentar", "Ver nubes", "Volver", "Si", "No", "Ver Comprimida"]
    return commands[pos] if pos < len(commands) else "Desconocido"


class MultiImageWindow(QMainWindow):
    """Ventana secundaria para mostrar múltiples imágenes procesadas."""
    def __init__(self, images_with_labels, title="Imágenes Comprimidas"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 900, 500)
        self.images_with_labels = images_with_labels
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        for image_path, label in self.images_with_labels:
            vbox = QVBoxLayout()

            # Mostrar la imagen
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setScaledContents(True)
                image_label.setFixedSize(250, 250)  # Ajusta el tamaño según sea necesario
                vbox.addWidget(image_label)
            else:
                vbox.addWidget(QLabel("No se pudo cargar la imagen."))

            # Mostrar la etiqueta de porcentaje
            label_widget = QLabel(label)
            label_widget.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label_widget)

            layout.addLayout(vbox)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class SecondaryWindow(QMainWindow):
    """Ventana secundaria para mostrar una imagen procesada (segmentada o comprimida)."""
    def __init__(self, image_path, title="Imagen Procesada"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 600, 400)
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Mostrar la imagen procesada
        if self.image_path and os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setScaledContents(True)
            layout.addWidget(image_label)
        else:
            layout.addWidget(QLabel("No se pudo cargar la imagen procesada."))

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Voz - Menú")
        self.setGeometry(100, 100, 600, 400)

        self.image_loaded = False  # Bandera para verificar si hay una imagen cargada
        self.state = "main_menu"  # Estados: main_menu, waiting_confirmation, waiting_3
        self.secondary_windows = []  # Lista para manejar ventanas secundarias
        self.image_path = None  # Inicializar atributo para la ruta de la imagen
        self.processed_image_path = None  # Ruta para imagen procesada
        self.last_action = None  # Última acción realizada: "segmentar" o "comprimir"

        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.start_recording_cycle)
        self.timer.start(4000)  # Ciclo de grabación con pausa cada 4 segundos

    def initUI(self):
        """Crea la interfaz gráfica."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.info_label = QLabel("Estado: Esperando imagen.")
        layout.addWidget(self.info_label)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Cargar Imagen")
        self.upload_button.clicked.connect(self.load_image)
        layout.addWidget(self.upload_button)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def load_image(self):
        """Carga una imagen en la aplicación."""
        image_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecciona una imagen", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_loaded = True
            self.info_label.setText("Estado: Imagen cargada. Esperando comandos (Segmentar, Comprimir).")
            self.image_path = image_path  # Almacena la ruta de la imagen

    def start_recording_cycle(self):
        """Inicia el ciclo de grabación con una pausa antes de grabar."""
        self.status_bar.showMessage("Haciendo pausa...")
        QTimer.singleShot(2000, self.start_audio_processing)  # Espera 2 segundos antes de procesar

    def start_audio_processing(self):
        """Inicia el procesamiento de audio en un hilo separado."""
        self.status_bar.showMessage("Grabando...")
        self.audio_thread = AudioProcessor()
        self.audio_thread.command_detected.connect(self.process_command)
        self.audio_thread.start()

    def segment_image(self, image_path):
        """Segmenta una imagen y guarda el resultado."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"La imagen no existe en la ruta: {image_path}")

            # Leer la imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo leer la imagen.")

            # Convertir a RGB y a un array 2D
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)

            # Configuración de K-means
            k = 4  # Número de segmentos
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Reconstruir la imagen segmentada
            centers = np.uint8(centers)
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(image_rgb.shape)

            # Guardar la imagen segmentada temporalmente
            segmented_path = "imagen_segmentada.jpg"
            cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

            return segmented_path  # Devuelve la ruta de la imagen segmentada
        except Exception as e:
            print(f"Error al segmentar la imagen: {e}")
            return None

    def comprimir_imagen(self):
        """
        Comprime la imagen cargada con los porcentajes especificados y muestra las imágenes comprimidas
        en una única ventana auxiliar, una al lado de la otra con el porcentaje de compresión abajo.
        Utiliza porcentajes predeterminados: [15, 50, 70].
        """
        if self.image_path is not None:
            try:
                # Definir porcentajes de compresión predeterminados
                porcentajes_compresion = [15, 50, 70]
                
                # Leer la imagen en color
                image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError("No se pudo leer la imagen.")
                
                # Aplicar la Transformada Discreta del Coseno (DCT) tipo 2
                dct_tipo2 = dct(dct(image.T, type=2, norm='ortho').T, type=2, norm='ortho')
                
                # Listas para almacenar las imágenes comprimidas y descomprimidas
                imagenes_comprimidas = []
                imagenes_descomprimidas = []
                
                # Lista para almacenar las imágenes comprimidas y sus etiquetas
                images_with_labels = []
                
                for porcentaje in porcentajes_compresion:
                    # Calcular la cantidad de coeficientes a conservar
                    total_coeffs = dct_tipo2.size
                    coeffs_to_keep = int(total_coeffs * (porcentaje / 100))
                    
                    # Crear una máscara de ceros
                    mask = np.zeros(dct_tipo2.shape, dtype=bool)
                    
                    # Obtener los índices de los coeficientes ordenados por magnitud descendente
                    sorted_indices = np.argsort(np.abs(dct_tipo2), axis=None)[::-1]
                    
                    # Conservar los primeros 'coeffs_to_keep' coeficientes
                    mask[np.unravel_index(sorted_indices[:coeffs_to_keep], dct_tipo2.shape)] = True
                    
                    # Aplicar la máscara
                    dct_comprimida = np.where(mask, dct_tipo2, 0)
                    
                    # Aplicar la IDCT para descomprimir la imagen
                    idct_tipo2 = idct(idct(dct_comprimida.T, type=2, norm='ortho').T, type=2, norm='ortho')
                    img_descomprimida = np.clip(idct_tipo2, 0, 255).astype(np.uint8)
                    
                    # Guardar las imágenes comprimidas y descomprimidas temporalmente
                    nombre_comprimida = f"imagen_comprimida_{porcentaje}.jpg"
                    nombre_descomprimida = f"imagen_descomprimida_{porcentaje}.jpg"
                    
                    cv2.imwrite(nombre_comprimida, dct_comprimida)
                    cv2.imwrite(nombre_descomprimida, img_descomprimida)
                    
                    imagenes_comprimidas.append((nombre_comprimida, porcentaje))
                    imagenes_descomprimidas.append((nombre_descomprimida, porcentaje))
                    
                    # Añadir la imagen comprimida y su etiqueta a la lista
                    images_with_labels.append((nombre_comprimida, f"{porcentaje}% de Compresión"))
                
                # Mostrar las imágenes comprimidas en una única ventana auxiliar
                self.open_multi_image_window(images_with_labels, title="Imágenes Comprimidas")
                
                # Actualizar el estado y la interfaz
                self.processed_image_path = imagenes_comprimidas[0][0]  # Por ejemplo, la primera imagen comprimida
                self.last_action = "comprimir"
                self.state = "waiting_3"  # Cambiar a 'waiting_3' para esperar cualquier comando como 'volver'
                self.info_label.setText("Compresión completada. ¿Deseas realizar otra acción? (Cualquier comando para volver)")
            
            except ValueError as ve:
                QMessageBox.critical(self, "Error", str(ve))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al comprimir la imagen: {str(e)}")
        else:
            QMessageBox.warning(self, "Advertencia", "Por favor, carga una imagen antes de comprimirla.")

    def open_multi_image_window(self, images_with_labels, title="Imágenes Comprimidas"):
        """Abre una ventana secundaria con múltiples imágenes procesadas."""
        window = MultiImageWindow(images_with_labels, title)
        self.secondary_windows.append(window)
        window.show()

    def process_command(self, command):
        """Procesa el comando detectado."""
        self.status_bar.showMessage(f"Comando detectado: {command}")

        if self.state == "waiting_3":
            # Cualquier comando recibido se trata como 'volver'
            self.info_label.setText("Cerrando ventanas y regresando al menú principal.")
            self.close_secondary_windows()
            self.state = "main_menu"
            return  # Salir para no procesar otros estados

        if self.state == "main_menu":
            if command.lower() == "segmentar":
                if self.image_loaded:
                    self.info_label.setText("Procesando segmentación...")
                    segmented_path = self.segment_image(self.image_path)  # Usar la ruta correcta
                    if segmented_path:
                        self.info_label.setText("Segmentación completada. ¿Deseas ver la imagen segmentada? (Si/No)")
                        self.processed_image_path = segmented_path  # Guardar la ruta de la imagen segmentada
                        self.last_action = "segmentar"
                        self.state = "waiting_confirmation"
                    else:
                        self.info_label.setText("Error al segmentar la imagen.")
                else:
                    self.info_label.setText("Por favor, carga una imagen antes de segmentar.")
        
            elif command.lower() == "comprimir":
                if self.image_loaded:
                    self.info_label.setText("Procesando compresión...")
                    self.comprimir_imagen()  # Usar el método de compresión actualizado
                else:
                    self.info_label.setText("Por favor, carga una imagen antes de comprimirla.")

        elif self.state == "waiting_confirmation":
            if command.lower() == "si":
                if self.last_action == "segmentar":
                    title = "Imagen Segmentada"
                    image_path = self.processed_image_path
                    self.open_secondary_window(image_path, title=title)
                elif self.last_action == "comprimir":
                    title = "Imágenes Comprimidas y Descomprimidas"
                    # Las imágenes comprimidas ya se han mostrado en una ventana auxiliar
                    QMessageBox.information(self, "Información", "Las imágenes comprimidas ya están abiertas en una ventana auxiliar.")
                self.state = "waiting_3" if self.last_action == "segmentar" else "main_menu"
                self.info_label.setText("¿Deseas realizar otra acción? (Cualquier comando para volver)")
            elif command.lower() == "no":
                self.info_label.setText("Acción cancelada. Regresando al menú principal.")
                self.state = "main_menu"

    def open_secondary_window(self, image_path, title="Imagen Procesada"):
        """Abre una nueva ventana secundaria con la imagen procesada."""
        if image_path and os.path.exists(image_path):
            window = SecondaryWindow(image_path, title)
            self.secondary_windows.append(window)
            window.show()
        else:
            QMessageBox.warning(self, "Advertencia", "No hay imagen procesada para mostrar.")

    def close_secondary_windows(self):
        """Cierra todas las ventanas secundarias."""
        for window in self.secondary_windows:
            window.close()
        self.secondary_windows.clear()

    def calcular_porcentaje(self, dct_matrix, porcentaje):
        """
        Calcula la cantidad de coeficientes de la DCT a conservar según el porcentaje.
        Retorna una máscara y una matriz DCT comprimida.
        """
        total_coeffs = dct_matrix.size
        coeffs_to_keep = int(total_coeffs * (porcentaje / 100))

        # Crear una máscara de ceros
        mask = np.zeros(dct_matrix.shape, dtype=bool)

        # Obtener los índices de los coeficientes ordenados por magnitud descendente
        sorted_indices = np.argsort(np.abs(dct_matrix), axis=None)[::-1]

        # Conservar los primeros 'coeffs_to_keep' coeficientes
        mask[np.unravel_index(sorted_indices[:coeffs_to_keep], dct_matrix.shape)] = True

        # Aplicar la máscara
        dct_comprimida = np.where(mask, dct_matrix, 0)

        return mask, dct_comprimida

    def intvalue(self, value):
        """Convierte un valor a entero y valida que esté entre 0 y 100."""
        try:
            val = int(value)
            if val < 0 or val > 100:
                raise ValueError
            return val
        except:
            raise ValueError("El porcentaje debe ser un número entero entre 0 y 100.")

    def mostrar_imagenes(self, imagenes):
        """
        Muestra las imágenes comprimidas en ventanas secundarias.
        Cada imagen se mostrará en una ventana separada con un título adecuado.
        """
        nombres = [
            "imagen_comprimida_input.jpg",
            "imagen_comprimida_80.jpg",
            "imagen_comprimida_50.jpg",
            "imagen_comprimida_25.jpg"
        ]
        titulos = [
            "Imagen Comprimida 100%",
            "Imagen Comprimida 80%",
            "Imagen Comprimida 50%",
            "Imagen Comprimida 25%"
        ]

        for img, nombre, titulo in zip(imagenes, nombres, titulos):
            # Guardar la imagen
            cv2.imwrite(nombre, img)

            # Crear y mostrar una ventana secundaria para cada imagen
            window = SecondaryWindow(nombre, title=titulo)
            self.secondary_windows.append(window)
            window.show()


def calculing(archivo):
    """Calcula la energía de una señal de audio."""
    try:
        muestreo, sonido = waves.read(archivo)
        if len(sonido.shape) > 1:
            xn = sonido[:, 0]
        else:
            xn = sonido
        energia = np.sum(np.abs(xn.astype(np.float64)) ** 2)
        return xn, energia
    except Exception as e:
        print(f"Error al calcular energía: {e}")
        return np.array([]), 0


def absFft(array):
    return np.abs(array)


def separate(array, num_blocks=32):
    if len(array) < num_blocks:
        num_blocks = len(array)
    segment_len = len(array) // num_blocks
    return [np.mean(array[i * segment_len:(i + 1) * segment_len]) for i in range(num_blocks)]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
