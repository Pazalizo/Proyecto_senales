import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import numpy as np
from scipy.io import wavfile as waves
from scipy.fft import fft  # FFT más eficiente usando SciPy
import pickle
import os

from Bordes import bordes
from Comprimir import Comprimir
from figuras import figuras

# Configuración de audio
formato = pyaudio.paInt16
canales = 1  # Canal mono
rate = 44100  # Frecuencia de muestreo
chunk = 1024  # Tamaño del buffer
duracion = 2  # Duración de la grabación en segundos
archivo = "señal.wav"  # Archivo donde se guarda el audio grabado


def record():
    audio = pyaudio.PyAudio()

    # Selecciona el dispositivo con índice 1
    try:
        stream = audio.open(format=formato, channels=canales, rate=rate, 
                            input=True, input_device_index=1, frames_per_buffer=chunk)
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

    # Guarda el audio grabado en un archivo WAV
    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(canales)
    waveFile.setsampwidth(audio.get_sample_size(formato))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def calculing(archivo):
    muestreo, sonido = waves.read(archivo)
    muestra = len(sonido)
    
    # Si el audio es estéreo, usa solo el primer canal
    if len(sonido.shape) > 1:
        xn = sonido[:, 0]
    else:
        xn = sonido

    # Calcula la energía de la señal
    energia = np.sum(np.abs(xn.astype(np.float64)) ** 2)
    return xn, energia


def absFft(array):
    return np.abs(array)


def separate(array, num_blocks=16):
    """
    Divide una señal en bloques iguales y calcula el promedio de cada bloque.
    """
    segment_len = len(array) // num_blocks
    return [
        np.mean(array[i * segment_len:(i + 1) * segment_len]) for i in range(num_blocks)
    ]


def cal_energia(array):
    """
    Calcula la energía en 4 bandas de la señal.
    """
    band_size = len(array) // 4
    return [
        np.sum(np.abs(array[i * band_size:(i + 1) * band_size]) ** 2)
        for i in range(4)
    ]


def substract(arr, arr2):
    """
    Resta elemento a elemento entre dos arrays.
    """
    return np.sum(np.abs(np.array(arr) - np.array(arr2)))


def comparing(array):
    """
    Encuentra el índice del menor valor en un array.
    """
    return np.argmin(array)


def prom(array):
    """
    Promedia los valores de varias listas para obtener una lista promedio.
    """
    return [np.mean([sub[i] for sub in array]) for i in range(len(array[0]))]


def start_processing():
    record()   
    xn, energia = calculing(archivo)
    fft_arr = fft(xn)  # Calcula la FFT usando SciPy
    fft_arr = absFft(fft_arr)
    rec = separate(fft_arr)

    print(f"Vector procesado: {rec}")

    banda_energies = cal_energia(rec)
    os.system('cls')
    print(f"Energía total: {energia}")
    print(f"Energías por banda: {banda_energies}")

    # Carga los datos procesados previamente
    with open('array_data.pkl', 'rb') as f:
        data = pickle.load(f)
    array, energias = data

    # Comparación
    compare = [substract(a, rec) for a in array]
    pos = comparing(compare)

    # Asigna la palabra reconocida
    if pos == 0:
        palabra = "Bordes"
    elif pos == 1:
        palabra = "Comprimir"
    elif pos == 2:
        palabra = "Figuras"
    else:
        palabra = "Desconocido"

    selected_option.set(f"Elegiste: {palabra}")


def run_algorithm():
    palabra = selected_option.get().split(": ")[1]

    if palabra == "Bordes":
        messagebox.showinfo("Información", "ALGORITMO DETECCION DE BORDES ACTIVADO")
        bordes(root)
    elif palabra == "Comprimir":
        messagebox.showinfo("Información", "ALGORITMO COMPRESION ACTIVADO")
        Comprimir(root)
    elif palabra == "Figuras":
        messagebox.showinfo("Información", "ALGORITMO RECONOCIMIENTO DE FIGURAS ACTIVADO")
        figuras(root)
    else:
        messagebox.showerror("Error", "Selección no válida")


def main(main):
    global selected_option
    global root

    main.destroy()

    root = tk.Tk()
    root.title("Reconocimiento de Voz")

    frame = tk.Frame(root)
    frame.pack(padx=95, pady=10)

    record_button = tk.Button(frame, text="Grabar y Procesar", command=start_processing)
    record_button.pack(pady=5)

    selected_option = tk.StringVar()
    selected_option.set("Graba un audio")

    result_label = tk.Label(frame, textvariable=selected_option)
    result_label.pack(pady=5)

    run_button = tk.Button(frame, text="Ejecutar Algoritmo", command=run_algorithm)
    run_button.pack(pady=5)

    root.mainloop()


# Lanza la aplicación principal
if __name__ == "__main__":
    root = tk.Tk()
    main(root)
