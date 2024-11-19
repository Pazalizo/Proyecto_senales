import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft

def plot_audio_spectrum(file_path, title, ax):
    # Cargar el archivo de audio
    sample_rate, data = wavfile.read(file_path)

    # Si el audio tiene más de un canal, tomar solo el primer canal
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalizar los datos de audio
    data = data / np.max(np.abs(data))

    # Calcular la FFT
    N = len(data)
    T = 1.0 / sample_rate
    yf = fft(data)
    xf = np.fft.fftfreq(N, T)[:N // 2]

    # Graficar el espectro
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), label=title)
    ax.axvline(x=700, color='red', linestyle='--', label="700 Hz")
    ax.axvline(x=2000, color='blue', linestyle='--', label="2000 Hz")
    ax.axvline(x=3000, color='green', linestyle='--', label="3000 Hz")
    ax.set_title(title)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    ax.grid()
    ax.set_xlim(0, 4000)  # Limitar el eje x hasta 4000 Hz
    ax.legend()

# Crear un gráfico con 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Archivo de ejemplo para cada categoría
files = {
    "Figuras": "./audios/figuras/Figuras2.wav",
    "Bordes": "./audios/bordes/Bordes2.wav",
    "Comprimir": "./audios/comprimir/Comprimir2.wav"
}

# Generar las gráficas
for idx, (category, file_path) in enumerate(files.items()):
    plot_audio_spectrum(file_path, f"Espectro de Frecuencia - {category}", axs[idx])

# Mostrar las gráficas
plt.tight_layout()
plt.show()
