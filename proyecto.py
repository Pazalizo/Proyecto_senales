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
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap

# Configuración de audio
formato = pyaudio.paInt16
canales = 1
rate = 44100
chunk = 1024
duracion = 2
archivo = "señal.wav"


def record():
    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=formato, channels=canales, rate=rate, 
            input=True, input_device_index=1, frames_per_buffer=chunk
        )
    except Exception as e:
        print(f"Error abriendo el dispositivo de audio: {e}")
        return

    print("\nGrabando...")
    frames = []

    for _ in range(0, int(rate / chunk * duracion)):
        data = stream.read(chunk)
        frames.append(data)

    print("\nGrabación terminada.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(canales)
    waveFile.setsampwidth(audio.get_sample_size(formato))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def calculing(archivo):
    muestreo, sonido = waves.read(archivo)
    if len(sonido.shape) > 1:
        xn = sonido[:, 0]
    else:
        xn = sonido
    energia = np.sum(np.abs(xn.astype(np.float64)) ** 2)
    return xn, energia


def absFft(array):
    return np.abs(array)


def separate(array, num_blocks=32):
    segment_len = len(array) // num_blocks
    return [np.mean(array[i * segment_len:(i + 1) * segment_len]) for i in range(num_blocks)]


def cal_energia(array):
    band_size = len(array) // 4
    return [np.sum(np.abs(array[i * band_size:(i + 1) * band_size]) ** 2) for i in range(4)]


def start_processing(selected_option_label):
    record()
    xn, energia = calculing(archivo)
    xn = xn / max(abs(xn))
    fft_arr = fft(xn)
    fft_arr = absFft(fft_arr)
    rec = separate(fft_arr)

    print(f"Vector procesado: {rec}")
    banda_energies = cal_energia(rec)

    with open('array_data.pkl', 'rb') as f:
        data = pickle.load(f)
    array, energias = data

    compare = [cosine(a, rec) for a in array]
    pos = compare.index(min(compare))

    if pos == 0:
        palabra = "Comprimir"
    elif pos == 1:
        palabra = "Segmentar"
    elif pos == 2:
        palabra = "Ver nubes"
    elif pos == 3:
        palabra = "Volver"
    elif pos == 4:
        palabra = "Si"
    elif pos == 5:
        palabra = "No"
    else:
        palabra = "Desconocido"

    selected_option_label.setText(f"Elegiste: {palabra}")
    print(selected_option_label.text())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Voz")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Botón para grabar y procesar
        self.record_button = QPushButton("Grabar y Procesar")
        self.record_button.clicked.connect(lambda: start_processing(self.selected_option_label))
        layout.addWidget(self.record_button)

        # Etiqueta para mostrar resultados
        self.selected_option_label = QLabel("Graba un audio")
        layout.addWidget(self.selected_option_label)

        # Botón para cargar imagen
        self.upload_button = QPushButton("Cargar Imagen")
        self.upload_button.clicked.connect(self.load_image)
        layout.addWidget(self.upload_button)

        # Etiqueta para mostrar imagen
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Selecciona una imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # Escalar la imagen para ajustarse a la etiqueta


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
