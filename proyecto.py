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
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QStatusBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


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
    xn = xn / max(abs(xn))
    fft_arr = fft(xn)
    fft_arr = absFft(fft_arr)
    rec = separate(fft_arr)

    with open('array_data.pkl', 'rb') as f:
        data = pickle.load(f)
    array, energias = data

    compare = [cosine(a, rec) for a in array]
    pos = compare.index(min(compare))

    commands = ["Comprimir", "Segmentar", "Ver nubes", "Volver", "Si", "No"]
    return commands[pos] if pos < len(commands) else "Desconocido"


class SecondaryWindow(QMainWindow):
    """Ventana secundaria que se abre al detectar 'Si'."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ventana Secundaria")
        self.setGeometry(200, 200, 400, 300)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("Esta es la nueva ventana.")
        layout.addWidget(label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Voz - Menú")
        self.setGeometry(100, 100, 600, 400)

        self.image_loaded = False  # Bandera para verificar si hay una imagen cargada
        self.state = "main_menu"  # Estados: main_menu, waiting_4_5, waiting_3
        self.secondary_windows = []  # Lista para manejar ventanas secundarias

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
        image_path, _ = QFileDialog.getOpenFileName(self, "Selecciona una imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_loaded = True
            self.info_label.setText("Estado: Imagen cargada. Esperando comandos (1, 2, 3).")

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

    def process_command(self, command):
        """Procesa el comando detectado."""
        self.status_bar.showMessage(f"Comando detectado: {command}")

        if self.state == "main_menu":
            if command in ["Comprimir", "Segmentar", "Ver nubes"]:
                self.info_label.setText(f"Comando {command} detectado. Esperando comando 'Si' (4) o 'No' (5).")
                self.state = "waiting_4_5"

        elif self.state == "waiting_4_5":
            if command == "Si":
                self.info_label.setText("Comando 'Si' detectado. Abriendo nueva ventana...")
                self.open_secondary_window()
                self.state = "waiting_3"
            elif command == "No":
                self.info_label.setText("Comando 'No' detectado. Regresando al menú principal.")
                self.state = "main_menu"

        elif self.state == "waiting_3":
            if command == "Volver":
                self.info_label.setText("Comando 'Volver' detectado. Cerrando ventanas y regresando al menú principal.")
                self.close_secondary_windows()
                self.state = "main_menu"

    def open_secondary_window(self):
        """Abre una nueva ventana secundaria."""
        window = SecondaryWindow()
        self.secondary_windows.append(window)
        window.show()

    def close_secondary_windows(self):
        """Cierra todas las ventanas secundarias."""
        for window in self.secondary_windows:
            window.close()
        self.secondary_windows.clear()


def calculing(archivo):
    """Calcula la energía de una señal de audio."""
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
