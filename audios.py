from math import pi
from scipy.fft import fft  # Usamos SciPy para FFT
import wave
import numpy as np
from scipy.io import wavfile as waves
import pickle
import os

def calculing(archivo):
    muestreo, sonido = waves.read(archivo)
    muestra = len(sonido)
    
    if len(sonido.shape) > 1:
        xn = sonido[:, 0]  # Estéreo: tomar solo el primer canal
    else:
        xn = sonido  # Mono: usar directamente la señal
    
    energia = np.sum(np.abs(xn.astype(np.float64) ** 2))
    potencia = energia / muestra

    return xn, energia


def absFft(array):
    return np.abs(array)

def separate(array, num_segments=32):
    arr = []
    segment_len = len(array) // num_segments
    for i in range(num_segments):
        segment = array[i * segment_len:(i + 1) * segment_len]
        arr.append(np.mean(segment))
    return arr

def prom(array):
    return [sum(sub[i] for sub in array) / len(array) for i in range(32)]


if __name__ == '__main__':
    array = []
    energia = []
    
    categorias = ["comprimir","segmentar","ver nubes","volver","si", "no"]
    for categoria in categorias:
        aux = []
        energy = []
        print(f"Procesando categoría: {categoria}")
        
        for i in range(20):  # Ajusta a 21 si tienes esa cantidad de archivos
            archivo = f"./audios/{categoria.capitalize()}/{categoria.capitalize()}{i+1}.wav"
            
            if not os.path.exists(archivo):
                print(f"Archivo no encontrado: {archivo}")
                continue
            
            print(f"Procesando archivo: {archivo}")
            try:
                xn, ener = calculing(archivo)
                fft_arr = fft(xn)
                fft_arr = absFft(fft_arr)
                aux.append(separate(fft_arr))
                energy.append(ener)
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")
        
        array.append(prom(aux))
        energia.append(np.mean(energy))

    datos = (array, energia)
    with open('array_data.pkl', 'wb') as f:
        pickle.dump(datos, f)

    print("Procesamiento completado. Resultados guardados en 'array_data.pkl'.")
